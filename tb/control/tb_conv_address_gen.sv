// conv_address_gen unit test against the golden-model golden vectors (build/m6/addr/<name>/
// {cfg.txt, rows.hex}; docs/m6_cnn_contract.md sec 7). For every vector:
//   run 1 with ready probability P1, run 2 immediately after (back to back, L14) with P2;
//   every generated (pad, row) is compared in order, no extra row may appear, the generator must
//   return to idle. Every 8th vector additionally aborts a run with a reset after a few rows and
//   then repeats the vector cleanly (reset-recovery check).
// +CASES=<file listing one vector dir per line>   +SEED=<n>   +MAXCASES=<n>

`timescale 1ns/1ps

module tb_conv_address_gen;
  localparam int unsigned MaxRows = 4096;

  logic        clk = 1'b0, rst_n = 1'b0;
  always #5 clk = ~clk;

  logic        cmd_valid = 1'b0, cmd_ready;
  logic [15:0] cmd_ub_row = '0;
  logic [3:0]  cmd_kh = '0, cmd_kw = '0;
  logic [11:0] cmd_oh0 = '0, cmd_ohcnt = '0;
  logic [15:0] in_w = '0, in_h = '0, out_w = '0;
  logic [3:0]  sh = '0, sw = '0;
  logic [7:0]  pad_t = '0, pad_l = '0;
  logic        row_valid, row_ready = 1'b0, row_pad, idle;
  logic [15:0] row_addr;

  conv_address_gen dut (
    .clk_i(clk), .rst_ni(rst_n),
    .cmd_valid_i(cmd_valid), .cmd_ready_o(cmd_ready), .cmd_ub_row_i(cmd_ub_row),
    .cmd_kh_i(cmd_kh), .cmd_kw_i(cmd_kw), .cmd_oh0_i(cmd_oh0), .cmd_ohcnt_i(cmd_ohcnt),
    .in_w_i(in_w), .in_h_i(in_h), .out_w_i(out_w), .sh_i(sh), .sw_i(sw),
    .pad_t_i(pad_t), .pad_l_i(pad_l),
    .row_valid_o(row_valid), .row_ready_i(row_ready), .row_pad_o(row_pad),
    .row_addr_o(row_addr), .idle_o(idle)
  );

  // expected rows and checker state
  logic [19:0] exp [MaxRows];
  int unsigned n_exp = 0, n_got = 0, errors = 0, checks = 0, rows_checked = 0;
  int unsigned ready_pct = 100;
  bit          checking = 1'b0;

  always @(posedge clk) begin
    if (rst_n && checking && row_valid && row_ready) begin
      if (n_got >= n_exp) begin
        errors++; $error("extra row beyond %0d: pad=%0b row=%h", n_exp, row_pad, row_addr);
      end else begin
        if ({row_pad, row_addr} !== exp[n_got]) begin
          errors++;
          $error("row %0d: got pad=%0b row=%h, expected pad=%0b row=%h", n_got,
                 row_pad, row_addr, exp[n_got][16], exp[n_got][15:0]);
        end
        rows_checked++;
      end
      n_got++;
    end
  end

  // random ready, re-drawn every cycle
  always @(posedge clk) begin
    #1;
    row_ready = ($urandom_range(0, 99) < ready_pct);
  end

  function automatic int get(input string key, input string names[$], input int vals[$]);
    foreach (names[i]) if (names[i] == key) return vals[i];
    $fatal(1, "cfg.txt: missing key %s", key);
    return 0;
  endfunction

  task automatic load_vector(input string dir);
    int fd, v, rc;
    string key;
    string names[$];
    int    vals[$];
    fd = $fopen({dir, "/cfg.txt"}, "r");
    if (fd == 0) $fatal(1, "cannot open %s/cfg.txt", dir);
    while ($fscanf(fd, "%s %d", key, v) == 2) begin names.push_back(key); vals.push_back(v); end
    $fclose(fd);
    in_w = 16'(get("in_w", names, vals));   in_h = 16'(get("in_h", names, vals));
    out_w = 16'(get("out_w", names, vals)); sh = 4'(get("sh", names, vals));
    sw = 4'(get("sw", names, vals));        pad_t = 8'(get("pad_t", names, vals));
    pad_l = 8'(get("pad_l", names, vals));  cmd_kh = 4'(get("kh", names, vals));
    cmd_kw = 4'(get("kw", names, vals));    cmd_oh0 = 12'(get("oh0", names, vals));
    cmd_ohcnt = 12'(get("oh_cnt", names, vals)); cmd_ub_row = 16'(get("ub_row", names, vals));
    n_exp = int'(cmd_ohcnt) * int'(out_w);
    if (n_exp > MaxRows) $fatal(1, "%s: %0d rows exceeds TB limit", dir, n_exp);
    for (int i = 0; i < MaxRows; i++) exp[i] = 'x;
    $readmemh({dir, "/rows.hex"}, exp);
    for (int i = 0; i < n_exp; i++)
      if (^exp[i] === 1'bx) $fatal(1, "%s: rows.hex has fewer than %0d rows", dir, n_exp);
    rc = 0;
  endtask

  // one full run of the loaded vector; abort_after > 0 -> reset after that many accepted rows
  task automatic run_vector(input string name, input int unsigned pct, input int unsigned abort_after);
    int unsigned guard;
    ready_pct = pct;
    n_got = 0;
    @(posedge clk); #1;
    guard = 0;
    while (!cmd_ready && guard < 200000) begin @(posedge clk); #1; guard++; end
    if (!cmd_ready) begin errors++; $error("%s: generator never became ready", name); return; end
    cmd_valid = 1'b1;
    @(posedge clk); #1;
    cmd_valid = 1'b0;
    checking = 1'b1;
    guard = 0;
    while (n_got < n_exp && guard < 400000) begin
      @(posedge clk); #1; guard++;
      if (abort_after != 0 && n_got >= abort_after) begin
        rst_n = 1'b0; checking = 1'b0;
        repeat (3) @(posedge clk);
        #1 rst_n = 1'b1;
        repeat (2) @(posedge clk);
        if (!idle || row_valid) begin errors++; $error("%s: not idle after reset", name); end
        return;
      end
    end
    if (n_got != n_exp) begin
      errors++; $error("%s: only %0d/%0d rows before timeout", name, n_got, n_exp);
    end
    // no extra rows, back to idle
    repeat (40) @(posedge clk);
    #1;
    if (n_got != n_exp) begin errors++; $error("%s: %0d rows, expected %0d", name, n_got, n_exp); end
    if (!idle || row_valid) begin errors++; $error("%s: not idle at end (row_valid=%0b)", name, row_valid); end
    checking = 1'b0;
    checks++;
  endtask

  initial begin
    string cases_file, line, name;
    int fd, nc, maxc, seed;
    cases_file = "build/m6/addr/cases.txt";
    void'($value$plusargs("CASES=%s", cases_file));
    if ($value$plusargs("SEED=%d", seed)) void'($urandom(seed));
    maxc = 1 << 30;
    void'($value$plusargs("MAXCASES=%d", maxc));
    repeat (5) @(posedge clk);
    #1 rst_n = 1'b1;
    repeat (3) @(posedge clk);

    fd = $fopen(cases_file, "r");
    if (fd == 0) $fatal(1, "cannot open %s", cases_file);
    nc = 0;
    while (nc < maxc && $fgets(line, fd) > 0) begin
      // strip trailing newline / spaces
      while (line.len() > 0 && (line.getc(line.len()-1) == "\n" || line.getc(line.len()-1) == " "
             || line.getc(line.len()-1) == "\r")) line = line.substr(0, line.len()-2);
      if (line.len() == 0) continue;
      name = line;
      load_vector(name);
      if (nc % 8 == 3 && n_exp > 6) begin      // reset in the middle of a run, then clean runs
        run_vector({name, " (aborted)"}, 70, 3 + (nc % 5));
        load_vector(name);
      end
      run_vector({name, " run1"}, 100, 0);
      run_vector({name, " run2"}, 20 + (nc * 13) % 80, 0);    // back to back, random stalls
      nc++;
    end
    $fclose(fd);
    $display("tb_conv_address_gen: vectors=%0d runs=%0d rows_checked=%0d errors=%0d",
             nc, checks, rows_checked, errors);
    if (errors == 0 && nc > 0) $display("CONV_ADDR_GEN_PASS");
    else                       $display("CONV_ADDR_GEN_FAIL");
    $finish;
  end

  initial begin #2s $display("CONV_ADDR_GEN_FAIL timeout"); $finish; end
endmodule
