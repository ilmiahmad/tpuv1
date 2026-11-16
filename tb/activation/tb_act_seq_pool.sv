// tb_act_seq_pool: act_seq + real activation_unit + pool_unit against an ACC memory model and a UB
// write monitor, with an independent SystemVerilog reference for requantization + 2x2 pooling.
//
//   * pool 1 (max) / pool 2 (avg), POOL_W in {2,4,6,8,10}, POOL_H in {2,4,6}  (every combination)
//   * pool 0 (plain ACTIVATE) regression, random row counts
//   * commands sent back to back (the next ACTIVATE is accepted while the previous one still drains,
//     mixed pooled / non-pooled)
//   * random ACC read-grant stalls (acc_rd_gnt low 0 / 30 / 70 % of the cycles, i.e. the matrix
//     sequencer owning the read port)
//   * every scenario repeated twice without reset (L14); ACC data and parameters change per scenario
// The UB write port never waits (contract sec 4), so there are no output stalls at this level; pool_unit's
// own output backpressure is covered by tb_pool_unit.
// Plusargs: +SEED=<n>  +REPS=<n>
`timescale 1ns/1ps

module tb_act_seq_pool;
  localparam int unsigned N = 16;
  localparam time TCK = 10ns, TA = 2ns;

  logic clk = 1'b0, rst_n = 1'b0;
  always #(TCK/2) clk = ~clk;

  // ---- DUT: act_seq + activation_unit ----------------------------------------------------------
  logic              cmd_valid = 1'b0;
  wire               cmd_ready;
  logic [10:0]       cmd_acc_row = '0;
  logic [15:0]       cmd_ub_row = '0, cmd_rows = '0;
  logic [1:0]        cmd_qset = '0, cmd_pool = '0;
  logic [15:0]       cmd_pool_w = '0;
  wire               idle;
  logic              acc_gnt = 1'b1;
  wire               acc_rd_en;
  wire  [10:0]       acc_rd_addr;
  logic [N*32-1:0]   acc_rd_data = '0;
  wire               au_valid, au_ready, au_out_valid, au_out_ready, au_busy;
  wire  [N*32-1:0]   au_data;
  wire  [1:0]        au_qset;
  wire  [N*8-1:0]    au_out_data;
  wire               ub_wr_en, act_active;
  wire  [15:0]       ub_wr_addr;
  wire  [N*8-1:0]    ub_wr_data;

  logic              cfg_we = 1'b0;
  logic [15:0]       cfg_target = '0;
  logic [63:0]       cfg_value = '0;
  wire               cfg_err;

  act_seq #(.N(N)) i_seq (
    .clk_i(clk), .rst_ni(rst_n),
    .cmd_valid_i(cmd_valid), .cmd_ready_o(cmd_ready), .cmd_acc_row_i(cmd_acc_row), .cmd_ub_row_i(cmd_ub_row),
    .cmd_rows_i(cmd_rows), .cmd_qset_i(cmd_qset), .cmd_pool_i(cmd_pool), .cmd_pool_w_i(cmd_pool_w),
    .idle_o(idle),
    .acc_rd_gnt_i(acc_gnt), .acc_rd_en_o(acc_rd_en), .acc_rd_addr_o(acc_rd_addr), .acc_rd_data_i(acc_rd_data),
    .au_valid_o(au_valid), .au_ready_i(au_ready), .au_data_o(au_data), .au_qset_o(au_qset),
    .au_out_valid_i(au_out_valid), .au_out_ready_o(au_out_ready), .au_out_data_i(au_out_data), .au_busy_i(au_busy),
    .ub_wr_en_o(ub_wr_en), .ub_wr_addr_o(ub_wr_addr), .ub_wr_data_o(ub_wr_data),
    .act_active_o(act_active)
  );

  activation_unit #(.N(N)) i_au (
    .clk_i(clk), .rst_ni(rst_n),
    .cfg_we_i(cfg_we), .cfg_target_i(cfg_target), .cfg_value_i(cfg_value), .cfg_err_o(cfg_err),
    .in_valid_i(au_valid), .in_ready_o(au_ready), .in_data_i(au_data), .in_qset_i(au_qset),
    .out_valid_o(au_out_valid), .out_ready_i(au_out_ready), .out_data_o(au_out_data), .busy_o(au_busy)
  );

  // ---- ACC model: read latency 1 (data valid the cycle after the granted read) -------------------
  logic [N*32-1:0] acc_mem [2048];
  always @(posedge clk) if (acc_rd_en) acc_rd_data <= acc_mem[acc_rd_addr];

  int gnt_low_pct = 0;
  always @(posedge clk) begin
    #TA acc_gnt = (int'($urandom_range(0, 99)) >= gnt_low_pct);
  end

  // ---- parameter model ---------------------------------------------------------------------------
  logic signed [31:0] P_M0   [4][N];
  int                 P_SH   [4][N];
  logic signed [31:0] P_BIAS [4][N];
  int                 P_ZP   [4];
  int                 P_LO   [4];
  int                 P_HI   [4];

  // ---- reference ---------------------------------------------------------------------------------
  function automatic int rq_lane(input logic signed [31:0] acc, input int q, input int c);
    logic signed [31:0] x, a, y;
    logic        [31:0] mask, rem, thr, z;
    logic signed [63:0] ab, nudge;
    int left, right, r;
    x     = acc + P_BIAS[q][c];                                // wraps in 32 bits
    left  = (P_SH[q][c] > 0) ? P_SH[q][c] : 0;
    right = (P_SH[q][c] > 0) ? 0 : -P_SH[q][c];
    a     = x << left;
    if (a == 32'sh80000000 && P_M0[q][c] == 32'sh80000000) y = 32'sh7fffffff;
    else begin
      ab    = 64'(a) * 64'(P_M0[q][c]);
      nudge = (ab >= 0) ? 64'sd1073741824 : (64'sd1 - 64'sd1073741824);
      y     = 32'((ab + nudge) / 64'sd2147483648);              // truncating division
    end
    mask = (32'd1 << right) - 32'd1;
    rem  = y & mask;
    thr  = (mask >> 1) + (y[31] ? 32'd1 : 32'd0);
    z    = 32'((y >>> right)) + ((rem > thr) ? 32'd1 : 32'd0);
    z    = z + 32'(P_ZP[q]);
    r    = $signed(z);
    if (r < P_LO[q]) r = P_LO[q];
    if (r > P_HI[q]) r = P_HI[q];
    return r;
  endfunction

  function automatic logic [N*8-1:0] rq_row(input int acc_row, input int q);
    logic [N*8-1:0] o;
    for (int c = 0; c < N; c++) o[c*8 +: 8] = 8'(rq_lane($signed(acc_mem[acc_row][c*32 +: 32]), q, c));
    return o;
  endfunction

  function automatic int avg4(input int s);
    return (s >= 0) ? ((s + 2) / 4) : ((s - 2) / 4);
  endfunction

  // expected UB writes (in order) -----------------------------------------------------------------
  logic [15:0]    eq_addr [$];
  logic [N*8-1:0] eq_data [$];
  int unsigned    exp_total_rows = 0, act_cycles = 0, errors = 0, checks = 0, n_cmds = 0;
  task automatic fail(input string m); $error("%s", m); errors++; endtask

  task automatic predict(input int pool, input int pw, input int rows, input int acc_row, input int ub_row, input int q);
    if (pool == 0) begin
      for (int r = 0; r < rows; r++) begin
        eq_addr.push_back(16'(ub_row + r)); eq_data.push_back(rq_row(acc_row + r, q));
      end
    end else begin
      int ph;
      ph = rows / pw;
      for (int i = 0; i < ph/2; i++)
        for (int j = 0; j < pw/2; j++) begin
          logic [N*8-1:0] v [4];
          logic [N*8-1:0] o;
          v[0] = rq_row(acc_row + (2*i)*pw   + 2*j,   q);
          v[1] = rq_row(acc_row + (2*i)*pw   + 2*j+1, q);
          v[2] = rq_row(acc_row + (2*i+1)*pw + 2*j,   q);
          v[3] = rq_row(acc_row + (2*i+1)*pw + 2*j+1, q);
          for (int c = 0; c < N; c++) begin
            int m, s, t;
            m = $signed(v[0][c*8 +: 8]); s = 0;
            for (int k = 0; k < 4; k++) begin
              t = $signed(v[k][c*8 +: 8]);
              if (t > m) m = t;
              s += t;
            end
            o[c*8 +: 8] = (pool == 1) ? 8'(m) : 8'(avg4(s));
          end
          eq_addr.push_back(16'(ub_row + i*(pw/2) + j)); eq_data.push_back(o);
        end
    end
    exp_total_rows += rows;
  endtask

  // ---- monitors ------------------------------------------------------------------------------------
  always @(posedge clk) begin
    if (act_active) act_cycles++;
    if (cfg_err) fail("activation unit reported cfg_err for a legal parameter write");
    if (ub_wr_en) begin
      checks++;
      if (eq_addr.size() == 0) fail($sformatf("unexpected UB write @%0d", ub_wr_addr));
      else begin
        logic [15:0] ea; logic [N*8-1:0] ed;
        ea = eq_addr.pop_front(); ed = eq_data.pop_front();
        if (ub_wr_addr !== ea || ub_wr_data !== ed)
          fail($sformatf("UB write mismatch: addr got %0d exp %0d, data got %h exp %h", ub_wr_addr, ea, ub_wr_data, ed));
      end
    end
  end

  // ---- stimulus helpers ----------------------------------------------------------------------------
  function automatic logic [N*32-1:0] rnd_acc_row();
    logic [N*32-1:0] r;
    int mode;
    mode = int'($urandom_range(0, 9));
    for (int c = 0; c < N; c++) begin
      logic signed [31:0] v;
      v = $urandom();
      if (mode < 6)       v = v >>> int'($urandom_range(10, 14));      // typical conv sums
      else if (mode < 8)  v = v >>> int'($urandom_range(0, 9));        // large
      else if (mode == 8) v = (c % 2) ? 32'sh7fffffff : 32'sh80000000; // extremes
      else                v = $signed($urandom_range(0, 8)) - 4;       // tiny / ties
      r[c*32 +: 32] = v;
    end
    return r;
  endfunction

  task automatic program_params();
    // random per-channel parameters for all 4 sets; the pipeline must be idle
    for (int q = 0; q < 4; q++) begin
      for (int c = 0; c < N; c++) begin
        int sh;
        logic signed [31:0] m0, bias;
        m0   = 32'sh40000000 + $signed(32'($urandom_range(0, 32'h3fffffff)));      // [2^30, 2^31-1]
        sh   = (int'($urandom_range(0, 9)) < 7) ? -int'($urandom_range(6, 14)) : int'($urandom_range(0, 38)) - 31;
        if (sh > 7) sh = 7;
        bias = $signed($urandom_range(0, 32'h1ffff)) - 32'sh10000;
        P_M0[q][c] = m0; P_SH[q][c] = sh; P_BIAS[q][c] = bias;
        @(posedge clk); #TA cfg_we = 1; cfg_target = 16'h1000 | (q << 4) | c; cfg_value = {24'd0, 8'(sh), m0};
        @(posedge clk); #TA cfg_target = 16'h2000 | (q << 4) | c; cfg_value = {32'd0, bias};
        @(posedge clk); #TA cfg_we = 0;
      end
      case (q)
        0: begin P_LO[q] = -128; P_HI[q] = 127; end                 // no clamp
        1: begin P_LO[q] = 0;    P_HI[q] = 127; end                 // ReLU
        2: begin P_LO[q] = 0;    P_HI[q] = 100; end                 // ReLU6-like
        default: begin P_LO[q] = -int'($urandom_range(0, 128)); P_HI[q] = int'($urandom_range(0, 127)); end
      endcase
      P_ZP[q] = int'($urandom_range(0, 255)) - 128;
      @(posedge clk); #TA cfg_we = 1; cfg_target = 16'h3000 | q;
      cfg_value = {40'd0, 8'(P_HI[q]), 8'(P_LO[q]), 8'(P_ZP[q])};
      @(posedge clk); #TA cfg_we = 0;
    end
    repeat (4) @(posedge clk);
  endtask

  task automatic send(input int pool, input int pw, input int rows, input int acc_row, input int ub_row, input int q);
    predict(pool, pw, rows, acc_row, ub_row, q);
    cmd_valid = 1; cmd_pool = 2'(pool); cmd_pool_w = 16'(pw); cmd_rows = 16'(rows);
    cmd_acc_row = 11'(acc_row); cmd_ub_row = 16'(ub_row); cmd_qset = 2'(q);
    do @(posedge clk); while (!cmd_ready);                          // pre-edge sample
    #TA cmd_valid = 0; n_cmds++;
  endtask

  task automatic wait_drain(input string name);
    int to;
    to = 0;
    while ((!idle || eq_addr.size() != 0) && to < 200000) begin @(posedge clk); to++; end
    repeat (4) @(posedge clk);
    if (!idle) fail($sformatf("%s: act_seq not idle at the end", name));
    if (eq_addr.size() != 0) fail($sformatf("%s: %0d expected UB writes never happened", name, eq_addr.size()));
    if (act_cycles != exp_total_rows) fail($sformatf("%s: act_active counted %0d cycles, expected %0d rows", name, act_cycles, exp_total_rows));
    act_cycles = 0; exp_total_rows = 0;
  endtask

  // ---------------------------------------------------------------------------------------------
  int unsigned reps = 2;
  initial begin
    int seed;
    int pw_list [5] = '{2, 4, 6, 8, 10};
    int ph_list [3] = '{2, 4, 6};
    seed = 1;
    if ($value$plusargs("SEED=%d", seed)) ;
    void'($urandom(seed));
    if ($value$plusargs("REPS=%d", reps)) ;
    repeat (5) @(posedge clk); #TA rst_n = 1'b1; repeat (5) @(posedge clk);

    for (int rep = 0; rep < reps; rep++) begin                      // L14: repeated without reset
      for (int stall = 0; stall < 3; stall++) begin
        string name;
        int cl [$][5];                                              // {pool, pw, rows, acc_row, ub_row}
        cl.delete();                                                // static in this block: clear per scenario
        gnt_low_pct = (stall == 0) ? 0 : (stall == 1) ? 30 : 70;
        name = $sformatf("rep%0d gnt_low=%0d%%", rep, gnt_low_pct);
        // new random ACC contents and parameters for every scenario (pipeline is idle here)
        for (int r = 0; r < 2048; r++) acc_mem[r] = rnd_acc_row();
        program_params();
        // build the command list
        for (int p = 1; p <= 2; p++)
          for (int a = 0; a < 5; a++)
            for (int b = 0; b < 3; b++) begin
              int pw, rows, accr, ubr;
              pw = pw_list[a]; rows = pw * ph_list[b];
              accr = int'($urandom_range(0, 2048 - rows));
              ubr  = int'($urandom_range(0, 65535 - rows/4));
              cl.push_back('{p, pw, rows, accr, ubr});
            end
        for (int k = 0; k < 24; k++) begin                          // plain ACTIVATE regression
          int rows, accr, ubr;
          rows = int'($urandom_range(1, 40));
          accr = int'($urandom_range(0, 2048 - rows));
          ubr  = int'($urandom_range(0, 65535 - rows));
          cl.push_back('{0, 0, rows, accr, ubr});
        end
        // shuffle (Fisher-Yates)
        for (int k = cl.size() - 1; k > 0; k--) begin
          int j; int t [5];
          j = int'($urandom_range(0, k));
          t = cl[k]; cl[k] = cl[j]; cl[j] = t;
        end
        @(posedge clk); #TA;                                        // start after an edge
        foreach (cl[k]) send(cl[k][0], cl[k][1], cl[k][2], cl[k][3], cl[k][4], int'($urandom_range(0, 3)));
        wait_drain(name);
        $display("[%s] %0d commands, UB writes checked so far: %0d, errors=%0d", name, cl.size(), checks, errors);
      end
    end

    $display("tb_act_seq_pool: commands=%0d ub_writes_checked=%0d errors=%0d", n_cmds, checks, errors);
    if (errors == 0) $display("M6_ACT_POOL_PASS"); else $display("M6_ACT_POOL_FAIL errors=%0d", errors);
    $finish;
  end

  initial begin #2000ms $display("M6_ACT_POOL_FAIL global timeout"); $finish; end
endmodule
