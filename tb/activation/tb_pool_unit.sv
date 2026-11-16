// tb_pool_unit: 2x2 pooling unit (docs/m6_cnn_contract.md sec 5).
//  S1  exhaustive average sums: every s in [-512,508] (all 1021) reaches every lane, checked against
//      C truncating division (s>=0 ? (s+2)/4 : (s-2)/4) written with SV signed '/' (truncates).
//  S2  random quads, extremes, ties, mixed modes per group, mode 0 pass-through, random bubbles
//      on the input and random stalls on the output.
//  S3  the golden-model vectors build/m6/pool/<name>/{in.hex,out.hex,meta.txt} (cases listed in
//      build/m6/pool/cases.txt, paths relative to the repo root): every vector TWICE back to back
//      (no reset, L14) under three bubble/stall settings; compared with both out.hex and the
//      in-TB reference.
// Plusargs: +SEED=<n>  +POOLDIR=<dir with cases.txt>  +NOVEC (skip S3)
`timescale 1ns/1ps

module tb_pool_unit;
  localparam int unsigned N    = 16;
  localparam int unsigned W    = N*8;
  localparam int unsigned MAXR = 4*8192;
  localparam time TCK = 10ns, TA = 2ns;

  logic clk = 1'b0, rst_n = 1'b0;
  always #(TCK/2) clk = ~clk;

  logic         in_valid = 1'b0, out_ready = 1'b1;
  wire          in_ready, out_valid, busy;
  logic [W-1:0] in_data = '0;
  logic [1:0]   in_mode = '0;
  wire  [W-1:0] out_data;

  pool_unit #(.N(N)) dut (
    .clk_i(clk), .rst_ni(rst_n), .in_valid_i(in_valid), .in_ready_o(in_ready), .in_data_i(in_data),
    .in_mode_i(in_mode), .out_valid_o(out_valid), .out_ready_i(out_ready), .out_data_o(out_data),
    .busy_o(busy)
  );

  // ---------------------------------------------------------------------------------------------
  // reference (independent of the RTL formulation)
  // ---------------------------------------------------------------------------------------------
  function automatic int ref_avg(input int s);
    return (s >= 0) ? ((s + 2) / 4) : ((s - 2) / 4);            // SV '/' truncates toward zero
  endfunction

  function automatic logic signed [7:0] lane_i8(input logic [W-1:0] row, input int l);
    return row[l*8 +: 8];
  endfunction

  // pooled row for the 4 rows q[0..3] in `mode` (0 = pass-through of q[0])
  function automatic logic [W-1:0] ref_pool(input logic [W-1:0] q [4], input int mode);
    logic [W-1:0] r;
    for (int l = 0; l < N; l++) begin
      int m, s;
      m = lane_i8(q[0], l); s = 0;
      for (int k = 0; k < 4; k++) begin
        int v;
        v = lane_i8(q[k], l);
        if (v > m) m = v;
        s += v;
      end
      if (mode == 0)      r[l*8 +: 8] = q[0][l*8 +: 8];
      else if (mode == 2) r[l*8 +: 8] = 8'(ref_avg(s));
      else                r[l*8 +: 8] = 8'(m);
    end
    return r;
  endfunction

  // ---------------------------------------------------------------------------------------------
  // generic stream driver/monitor
  // ---------------------------------------------------------------------------------------------
  int unsigned errors = 0, checks = 0;
  task automatic fail(input string m); $error("%s", m); errors++; endtask

  logic [W-1:0] st_in   [MAXR];     // stimulus rows
  int           st_mode [MAXR];     // per-row mode (constant inside a group)
  logic [W-1:0] st_exp  [MAXR];     // expected pooled rows
  int           n_rows, n_exp;      // rows driven / pooled rows expected
  logic [W-1:0] got [MAXR];
  int           n_got;
  int           bubble_pct = 0, stall_pct = 0;

  // output side: random stalls + collection (pre-edge values: in the active region of the edge)
  int n_in_hs = 0;
  always @(posedge clk) begin
    if (in_valid && in_ready) begin
      if ($test$plusargs("DBG2") && n_in_hs < 8) $display("[%0t] hs#%0d in_data=%h st_in[%0d]=%h", $time, n_in_hs, in_data, n_in_hs, st_in[n_in_hs]);
      n_in_hs++;
    end
    if (out_valid && out_ready && n_got < MAXR) begin got[n_got] <= out_data; n_got <= n_got + 1; end
    #TA out_ready = (int'($urandom_range(0, 99)) >= stall_pct);
  end

  // optional trace: +DBG
  always @(posedge clk) if ($test$plusargs("DBG") && n_got > 1017 && n_got < 1030)
    $display("[%0t] in_v=%b in_r=%b mode=%0d out_v=%b out_r=%b busy=%b n_got=%0d", $time, in_valid, in_ready, in_mode, out_valid, out_ready, busy, n_got);

  task automatic run_stream(input string name);
    int r, to;
    @(posedge clk); #TA;                 // start strictly after an edge (no same-timestep race)
    n_got = 0; n_in_hs = 0;
    r = 0;
    while (r < n_rows) begin
      if (int'($urandom_range(0, 99)) < bubble_pct) begin
        in_valid = 1'b0; @(posedge clk); #TA;
      end else begin
        in_valid = 1'b1; in_data = st_in[r]; in_mode = st_mode[r][1:0];
        do @(posedge clk); while (!(in_valid && in_ready));     // pre-edge sample of in_ready
        #TA in_valid = 1'b0; r++;
      end
    end
    in_valid = 1'b0;
    to = 0;
    while ((n_got < n_exp || busy) && to < 4000) begin @(posedge clk); to++; end
    repeat (3) @(posedge clk);
    if (n_got != n_exp) fail($sformatf("%s: got %0d pooled rows, expected %0d (rows driven %0d, accepted %0d)", name, n_got, n_exp, n_rows, n_in_hs));
    else begin
      for (int g = 0; g < n_exp; g++) begin
        checks++;
        if (got[g] !== st_exp[g]) begin
          fail($sformatf("%s: group %0d: got %h expected %h", name, g, got[g], st_exp[g]));
          break;
        end
      end
    end
    if (busy) fail($sformatf("%s: busy after the stream", name));
  endtask

  // build the expected rows of the current stimulus (groups of 4, or 1 row when mode==0)
  task automatic build_exp();
    int r;
    r = 0; n_exp = 0;
    while (r < n_rows) begin
      logic [W-1:0] q [4];
      if (st_mode[r] == 0) begin
        q[0] = st_in[r]; q[1] = '0; q[2] = '0; q[3] = '0;
        st_exp[n_exp++] = ref_pool(q, 0); r += 1;
      end else begin
        for (int k = 0; k < 4; k++) q[k] = st_in[r+k];
        st_exp[n_exp++] = ref_pool(q, st_mode[r]); r += 4;
      end
    end
  endtask

  function automatic logic [W-1:0] rnd_row();
    logic [W-1:0] r;
    for (int i = 0; i < W/32; i++) r[i*32 +: 32] = $urandom();
    return r;
  endfunction

  // ---------------------------------------------------------------------------------------------
  // S1/S2 helpers
  // ---------------------------------------------------------------------------------------------
  // four int8 values with a given sum (feasible by construction)
  task automatic make_quad(input int s, output int v [4]);
    int rem;
    rem = s;
    for (int k = 0; k < 4; k++) begin
      int lo, hi, left;
      left = 3 - k;
      lo = (rem - 127*left > -128) ? rem - 127*left : -128;
      hi = (rem + 128*left < 127)  ? rem + 128*left : 127;
      v[k] = lo + int'($urandom_range(0, hi - lo));
      rem -= v[k];
    end
  endtask

  task automatic s1_sums();
    n_rows = 0;
    for (int t = 0; t < 1021; t++) begin
      int q [N][4];
      for (int l = 0; l < N; l++) begin
        int v [4]; int sum;
        sum = -512 + ((t + l*61) % 1021);
        make_quad(sum, v);
        for (int k = 0; k < 4; k++) q[l][k] = v[k];
      end
      for (int k = 0; k < 4; k++) begin
        logic [W-1:0] row;
        for (int l = 0; l < N; l++) row[l*8 +: 8] = 8'(q[l][k]);
        st_in[n_rows] = row; st_mode[n_rows] = 2; n_rows++;
      end
    end
    build_exp();
    // extra check: the reference of build_exp IS the C-division formula; assert it on the sums too
    for (int t = 0; t < 1021; t++) begin
      int s;
      s = -512 + t;
      if (ref_avg(s) < -128 || ref_avg(s) > 127) fail($sformatf("avg(%0d) out of int8", s));
    end
    bubble_pct = 0; stall_pct = 0;  run_stream("S1 all sums (no stalls)");
    bubble_pct = 40; stall_pct = 40; run_stream("S1 all sums (stalls)");
  endtask

  task automatic s2_random();
    // mixed modes per group, incl. pass-through rows, extremes and ties
    for (int pass = 0; pass < 3; pass++) begin
      n_rows = 0;
      for (int g = 0; g < 600; g++) begin
        int mode;
        mode = int'($urandom_range(0, 9));
        mode = (mode < 4) ? 1 : (mode < 8) ? 2 : (mode < 9) ? 0 : 3;     // 3 behaves as max
        if (mode == 0) begin
          st_in[n_rows] = rnd_row(); st_mode[n_rows] = 0; n_rows++;
        end else begin
          int kind;
          kind = int'($urandom_range(0, 5));
          for (int k = 0; k < 4; k++) begin
            logic [W-1:0] row;
            row = rnd_row();
            if (kind == 1) row = {W/8{8'h80}};                          // all -128
            if (kind == 2) row = {W/8{8'h7f}};                          // all 127
            if (kind == 3) row = st_in[(n_rows >= k) ? n_rows - k : 0] ^ ((k == 0) ? '0 : '0); // ties-ish
            if (kind == 4 && k != 2) row = {W/8{8'h80}};                // mostly -128
            st_in[n_rows] = row; st_mode[n_rows] = mode; n_rows++;
          end
        end
      end
      // mode 3 is "max" by definition (undefined in the contract; the decoder rejects it)
      for (int r = 0; r < n_rows; r++) if (st_mode[r] == 3) st_mode[r] = 1;
      build_exp();
      bubble_pct = (pass == 0) ? 0 : 50; stall_pct = (pass == 2) ? 70 : (pass == 1 ? 30 : 0);
      run_stream($sformatf("S2 random mixed modes pass %0d", pass));
    end
  endtask

  // ---------------------------------------------------------------------------------------------
  // S3: the golden-model vectors
  // ---------------------------------------------------------------------------------------------
  logic [W-1:0] vin  [MAXR];
  logic [W-1:0] vout [MAXR];

  function automatic int count_hex_lines(input string path);
    int fd, n, rc;
    logic [W-1:0] x;
    fd = $fopen(path, "r");
    if (fd == 0) return -1;
    n = 0;
    while (!$feof(fd)) begin
      rc = $fscanf(fd, "%h\n", x);
      if (rc == 1) n++;
    end
    $fclose(fd);
    return n;
  endfunction

  task automatic read_hex(input string path, input int n, ref logic [W-1:0] arr [MAXR]);
    int fd, rc, i;
    logic [W-1:0] x;
    fd = $fopen(path, "r"); i = 0;
    while (!$feof(fd) && i < n) begin
      rc = $fscanf(fd, "%h\n", x);
      if (rc == 1) begin arr[i] = x; i++; end
    end
    $fclose(fd);
  endtask

  // meta.txt: "key value" lines; mode = max|avg|1|2
  task automatic read_meta(input string path, output int mode);
    int fd, rc;
    string key, val;
    mode = -1;
    fd = $fopen(path, "r");
    if (fd == 0) return;
    while (!$feof(fd)) begin
      rc = $fscanf(fd, "%s %s\n", key, val);
      if (rc == 2 && key == "mode") begin
        if (val == "max" || val == "1") mode = 1;
        else if (val == "avg" || val == "2") mode = 2;
        else mode = val.atoi();
      end
    end
    $fclose(fd);
  endtask

  task automatic s3_vectors(input string pooldir);
    int fd, rc, ncases;
    string dir, line;
    fd = $fopen({pooldir, "/cases.txt"}, "r");
    if (fd == 0) begin
      $display("[NOTE] %s/cases.txt not found: S3 skipped", pooldir);
      return;
    end
    ncases = 0;
    while (!$feof(fd)) begin
      rc = $fscanf(fd, "%s\n", dir);
      if (rc != 1) continue;
      begin
        int mode, nin, nout, ok_in, ok_out;
        read_meta({dir, "/meta.txt"}, mode);
        nin  = count_hex_lines({dir, "/in.hex"});
        nout = count_hex_lines({dir, "/out.hex"});
        if (mode < 0 || nin <= 0 || nout <= 0 || nin != 4 * nout) begin
          fail($sformatf("vector %s: bad meta/shape (mode=%0d in=%0d out=%0d)", dir, mode, nin, nout));
          continue;
        end
        read_hex({dir, "/in.hex"},  nin,  vin);
        read_hex({dir, "/out.hex"}, nout, vout);
        for (int setting = 0; setting < 3; setting++) begin
          bubble_pct = (setting == 0) ? 0 : (setting == 1) ? 35 : 80;
          stall_pct  = (setting == 0) ? 0 : (setting == 1) ? 35 : 80;
          for (int rep = 0; rep < 2; rep++) begin           // twice back to back (L14)
            n_rows = nin;
            for (int r = 0; r < nin; r++) begin st_in[r] = vin[r]; st_mode[r] = mode; end
            build_exp();
            for (int g = 0; g < nout; g++)                  // out.hex must equal the in-TB reference
              if (st_exp[g] !== vout[g]) begin
                fail($sformatf("vector %s: out.hex group %0d disagrees with the in-TB reference", dir, g));
                break;
              end
            for (int g = 0; g < nout; g++) st_exp[g] = vout[g];   // judge the DUT against B's file
            run_stream($sformatf("S3 %s set%0d rep%0d", dir, setting, rep));
          end
        end
        ncases++;
        $display("[PASS?] vector %s: mode=%0d groups=%0d done", dir, mode, nout);
      end
    end
    $fclose(fd);
    $display("S3: %0d vector sets exercised", ncases);
  endtask

  // ---------------------------------------------------------------------------------------------
  initial begin
    int seed;
    string pooldir;
    seed = 1;
    if ($value$plusargs("SEED=%d", seed)) ;
    void'($urandom(seed));
    pooldir = "build/m6/pool";
    if ($value$plusargs("POOLDIR=%s", pooldir)) ;
    n_got = 0; n_rows = 0; n_exp = 0;
    repeat (4) @(posedge clk); #TA rst_n = 1'b1; repeat (3) @(posedge clk);

    s1_sums();
    s2_random();
    // S2 once more without reset (state must stay clean)
    s2_random();
    if (!$test$plusargs("NOVEC")) s3_vectors(pooldir);

    $display("tb_pool_unit: checks=%0d errors=%0d", checks, errors);
    if (errors == 0) $display("M6_POOL_UNIT_PASS"); else $display("M6_POOL_UNIT_FAIL errors=%0d", errors);
    $finish;
  end

  initial begin #400ms $display("M6_POOL_UNIT_FAIL global timeout"); $finish; end
endmodule
