// M4 activation_unit testbench (docs/m4_compute_core.md sec 9 A).
// Vectors: build/m4/requant/ from model/activation/gen_requant_vectors.py.
//
// Sequence (run twice back-to-back without reset, L14, then once more after a mid-stream reset):
//   P0: load params_p0 via cfg_we (SET_CONFIG semantics), stream in_p0, check out_p0
//   P1: while the last P0 rows are still in the (stalled) pipeline, load params_p1 -> those
//       rows must still give P0 results ("affects only rows entering after it"); the first P1
//       row is accepted in the cycle right after the last write. Stream in_p1, check out_p1.
//   BAD: replay bad_params (each must pulse cfg_err_o the next cycle), re-stream in_p1,
//       expect out_p1 unchanged.
// Random in_valid gaps (+IN_GAP_PCT) and out_ready stalls (+OUT_STALL_PCT). An independent SV
// reference re-derives every expected row from params/in and must agree with out_p<k>.hex.

`timescale 1ns/1ps

module tb_activation_unit;
  localparam int unsigned N       = 16;
  localparam int unsigned MAXROWS = 4096;
  localparam int unsigned MAXPAR  = 256;
  localparam time         TCK = 10ns, TA = 2ns;

  logic clk = 1'b0, rst_n = 1'b0;
  always #(TCK/2) clk = ~clk;

  logic              cfg_we = 1'b0;
  logic [15:0]       cfg_target = '0;
  logic [63:0]       cfg_value = '0;
  logic              cfg_err;
  logic              in_valid = 1'b0, in_ready;
  logic [N*32-1:0]   in_data = '0;
  logic [1:0]        in_qset = '0;
  logic              out_valid, out_ready = 1'b0;
  logic [N*8-1:0]    out_data;
  logic              busy;

  activation_unit #(.N(N)) dut (
    .clk_i(clk), .rst_ni(rst_n),
    .cfg_we_i(cfg_we), .cfg_target_i(cfg_target), .cfg_value_i(cfg_value), .cfg_err_o(cfg_err),
    .in_valid_i(in_valid), .in_ready_o(in_ready), .in_data_i(in_data), .in_qset_i(in_qset),
    .out_valid_o(out_valid), .out_ready_i(out_ready), .out_data_o(out_data), .busy_o(busy)
  );

  // ---------------- vectors ----------------
  logic [127:0]    par   [2][MAXPAR];
  logic [127:0]    bad   [MAXPAR];
  logic [N*32-1:0] vin   [2][MAXROWS];
  logic [3:0]      vq    [2][MAXROWS];
  logic [N*8-1:0]  vout  [2][MAXROWS];
  int unsigned     nrows [2], npar [2], nbad;
  logic [127:0]    t_par [MAXPAR];
  logic [N*32-1:0] t_in  [MAXROWS];
  logic [3:0]      t_q   [MAXROWS];
  logic [N*8-1:0]  t_out [MAXROWS];
  string           vdir = "build/m4/requant";
  int unsigned     seed = 1, in_gap = 30, out_stall = 30;

  function automatic int unsigned meta(string key);
    int fd, v, r; string k;
    fd = $fopen({vdir, "/meta.txt"}, "r");
    if (fd == 0) $fatal(1, "cannot open %s/meta.txt", vdir);
    while (!$feof(fd)) begin
      r = $fscanf(fd, "%s %d\n", k, v);
      if (r == 2 && k == key) begin $fclose(fd); return v; end
    end
    $fclose(fd);
    $fatal(1, "meta key %s missing", key);
    return 0;
  endfunction

  // ---------------- independent SV reference (contract sec 1) ----------------
  typedef struct packed {
    logic [31:0] m0, bias; logic [7:0] sh;
  } chp_t;
  chp_t       ref_ch  [4][N];
  logic [7:0] ref_zp  [4], ref_min [4], ref_max [4];

  function automatic void ref_write(logic [15:0] t, logic [63:0] v);
    if (t[15:12] == 4'h1 && t[11:6] == 0) begin ref_ch[t[5:4]][t[3:0]].m0 = v[31:0]; ref_ch[t[5:4]][t[3:0]].sh = v[39:32]; end
    else if (t[15:12] == 4'h2 && t[11:6] == 0) ref_ch[t[5:4]][t[3:0]].bias = v[31:0];
    else if (t[15:12] == 4'h3 && t[11:2] == 0) begin ref_zp[t[1:0]] = v[7:0]; ref_min[t[1:0]] = v[15:8]; ref_max[t[1:0]] = v[23:16]; end
  endfunction

  function automatic logic signed [31:0] ref_srdhm(logic signed [31:0] a, logic signed [31:0] b);
    logic signed [63:0] ab, v, q;
    if (a == 32'sh8000_0000 && b == 32'sh8000_0000) return 32'sh7FFF_FFFF;
    ab = 64'(a) * 64'(b);
    v  = ab + ((ab >= 0) ? 64'sd1073741824 : -64'sd1073741823);
    q  = (v >= 0) ? (v / 64'sd2147483648) : -((-v) / 64'sd2147483648);   // C truncation
    return q[31:0];
  endfunction

  function automatic logic signed [31:0] ref_rdbpot(logic signed [31:0] x, int e);
    logic [31:0] mask, rem, thr;
    mask = (32'd1 << e) - 1;
    rem  = 32'(x) & mask;
    thr  = (mask >> 1) + ((x < 0) ? 1 : 0);
    return (x >>> e) + ((rem > thr) ? 1 : 0);
  endfunction

  function automatic logic [N*8-1:0] ref_row(logic [N*32-1:0] acc, int q);
    logic [N*8-1:0] o;
    for (int c = 0; c < N; c++) begin
      logic signed [31:0] x, y, z; logic signed [7:0] sh; int left, right;
      x  = $signed(acc[c*32 +: 32]) + $signed(ref_ch[q][c].bias);
      sh = $signed(ref_ch[q][c].sh);
      left  = (sh > 0) ? sh : 0;
      right = (sh > 0) ? 0 : -sh;
      y = ref_rdbpot(ref_srdhm(x <<< left, $signed(ref_ch[q][c].m0)), right);
      z = y + 32'($signed(ref_zp[q]));
      if (z < $signed(ref_min[q]))      o[c*8 +: 8] = ref_min[q];
      else if (z > $signed(ref_max[q])) o[c*8 +: 8] = ref_max[q];
      else                              o[c*8 +: 8] = z[7:0];
    end
    return o;
  endfunction

  // ---------------- checking ----------------
  logic [N*8-1:0] exp_q[$];
  int unsigned    errors = 0, checked = 0, accepted = 0, cfgerr_seen = 0;

  always @(posedge clk) begin
    if (rst_n && out_valid && out_ready) begin
      if (exp_q.size() == 0) begin
        errors++; $error("unexpected output row %h", out_data);
      end else begin
        logic [N*8-1:0] e;
        e = exp_q.pop_front();
        if (out_data !== e) begin
          errors++;
          if (errors < 10) $error("row %0d: got %h exp %h", checked, out_data, e);
        end
        checked++;
      end
    end
    if (rst_n && in_valid && in_ready) accepted++;
    if (cfg_err) cfgerr_seen++;
  end

  // output stall pattern
  bit hold_out = 1'b0;
  always @(posedge clk) begin
    #TA out_ready = !hold_out && ($urandom_range(0, 99) >= out_stall);
  end

  // SVA
  a_out_hold: assert property (@(posedge clk) disable iff (!rst_n)
      (out_valid && !out_ready) |=> (out_valid && $stable(out_data)))
    else $error("output changed while stalled");
  a_in_ready: assert property (@(posedge clk) disable iff (!rst_n)
      out_ready |-> in_ready)
    else $error("in_ready low although out_ready high");

  // ---------------- drivers ----------------
  task automatic cfg_write(logic [127:0] w);
    @(posedge clk); #TA cfg_we = 1'b1; cfg_target = w[79:64]; cfg_value = w[63:0];
    ref_write(w[79:64], w[63:0]);
    @(posedge clk); #TA cfg_we = 1'b0;
  endtask

  // writes back-to-back (one per cycle), ending with cfg_we low at the cycle after the last
  task automatic cfg_burst(int k);
    for (int i = 0; i < npar[k]; i++) begin
      @(posedge clk); #TA cfg_we = 1'b1; cfg_target = par[k][i][79:64]; cfg_value = par[k][i][63:0];
      ref_write(par[k][i][79:64], par[k][i][63:0]);
    end
    @(posedge clk); #TA cfg_we = 1'b0;
  endtask

  task automatic stream(int k, bit gaps = 1'b1);
    for (int r = 0; r < nrows[k]; r++) begin
      logic [N*8-1:0] e;
      e = ref_row(vin[k][r], int'(vq[k][r]));
      if (e !== vout[k][r]) begin
        errors++; $error("SV reference disagrees with out_p%0d row %0d: sv=%h file=%h", k, r, e, vout[k][r]);
      end
      while (gaps && $urandom_range(0, 99) < in_gap) begin #TA in_valid = 1'b0; @(posedge clk); end
      #TA in_valid = 1'b1; in_data = vin[k][r]; in_qset = vq[k][r][1:0];
      exp_q.push_back(vout[k][r]);
      do @(posedge clk); while (!in_ready);
    end
    #TA in_valid = 1'b0;
  endtask

  task automatic drain();
    int unsigned n = 0;
    while ((exp_q.size() != 0 || busy) && n < 100000) begin @(posedge clk); n++; end
    if (exp_q.size() != 0) begin errors++; $error("drain timeout, %0d rows missing", exp_q.size()); end
  endtask

  task automatic one_pass(int pass);
    int unsigned e0;
    // P0
    cfg_burst(0);
    stream(0);
    // P1: freeze the output so the tail of P0 sits in the pipe, rewrite all params, then
    // stream P1 starting the cycle after the last write
    @(posedge clk); hold_out = 1'b1;
    repeat (3) @(posedge clk);
    cfg_burst(1);
    hold_out = 1'b0;
    stream(1);
    drain();
    // BAD writes: each must pulse cfg_err once and change nothing
    e0 = cfgerr_seen;
    for (int i = 0; i < nbad; i++) cfg_write(bad[i]);
    repeat (2) @(posedge clk);
    if (cfgerr_seen - e0 != nbad) begin
      errors++; $error("cfg_err pulses %0d, expected %0d", cfgerr_seen - e0, nbad);
    end
    stream(1);
    drain();
    $display("pass %0d done: checked=%0d errors=%0d", pass, checked, errors);
  endtask

  initial begin
    if ($value$plusargs("VEC=%s", vdir)) ;
    if ($value$plusargs("SEED=%d", seed)) ;
    if ($value$plusargs("IN_GAP_PCT=%d", in_gap)) ;
    if ($value$plusargs("OUT_STALL_PCT=%d", out_stall)) ;
    void'($urandom(seed));
    if (meta("N") != N) $fatal(1, "vector N mismatch");
    for (int k = 0; k < 2; k++) begin
      nrows[k] = meta($sformatf("rows_p%0d", k));
      npar[k]  = meta($sformatf("params_p%0d", k));
      if (nrows[k] > MAXROWS || npar[k] > MAXPAR) $fatal(1, "vector set too large");
    end
    nbad = meta("bad_params");
    // (Verilator cannot $readmemh into a slice of a 2-D array: load 1-D, then copy)
    for (int k = 0; k < 2; k++) begin
      $readmemh({vdir, $sformatf("/params_p%0d.hex", k)}, t_par);
      $readmemh({vdir, $sformatf("/in_p%0d.hex", k)},     t_in);
      $readmemh({vdir, $sformatf("/qset_p%0d.hex", k)},   t_q);
      $readmemh({vdir, $sformatf("/out_p%0d.hex", k)},    t_out);
      for (int i = 0; i < MAXPAR; i++)  par[k][i]  = t_par[i];
      for (int i = 0; i < MAXROWS; i++) begin vin[k][i] = t_in[i]; vq[k][i] = t_q[i]; vout[k][i] = t_out[i]; end
    end
    $readmemh({vdir, "/bad_params.hex"}, bad);
    $display("tb_activation_unit: N=%0d rows=%0d/%0d params=%0d/%0d bad=%0d seed=%0d gap=%0d%% stall=%0d%%",
             N, nrows[0], nrows[1], npar[0], npar[1], nbad, seed, in_gap, out_stall);

    repeat (5) @(posedge clk); #TA rst_n = 1'b1;
    one_pass(1);
    one_pass(2);                                   // back-to-back, no reset (L14)
    // mid-stream reset, then a clean pass (parameters reset to 0 in the DUT; reloaded by pass 3)
    cfg_burst(0);
    fork
      stream(0);
      begin repeat (200) @(posedge clk); #TA rst_n = 1'b0; end
    join_any
    disable fork;
    #TA in_valid = 1'b0; exp_q.delete();
    repeat (3) @(posedge clk); #TA rst_n = 1'b1;
    one_pass(3);

    $display("tb_activation_unit: accepted=%0d checked=%0d cfg_err_pulses=%0d errors=%0d",
             accepted, checked, cfgerr_seen, errors);
    if (errors == 0) $display("M4_ACT_PASS"); else $display("M4_ACT_FAIL errors=%0d", errors);
    $finish;
  end

  initial begin #50ms $display("M4_ACT_FAIL global timeout"); $finish; end

endmodule
