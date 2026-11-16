// M3 - matrix_unit testbench (Verilator + Xcelium). Contract: docs/m3_matrix_unit.md sec 5.
//
//   +VEC=<dir>         golden vector directory (meta.txt, x.hex, xnew.hex, w.hex, y.hex)
//   +BUBBLE_PCT=<p>    probability (%) of an x bubble per cycle            (default 20)
//   +WGAP_PCT=<p>      probability (%) the weight FIFO is not valid       (default 30)
//   +SEED=<s>
// Sequence: run 1, run 2 back-to-back (no reset, L14), run 3 aborted by a mid-run reset,
// run 4 after reset. Every output row compared bit-exactly, in order, at latency 2N.
// Checks (procedural so both simulators run them identically):
//   C1 x_ready low only on a valid new-weights row
//   C2 no weight shift in the 2N-2 cycles after a new-weights row is accepted
//   C3 one y row per accepted x row, exact latency L = 2N, nothing extra
//   C4 the golden y.hex agrees with an independent in-TB reference

`timescale 1ns/1ps

module tb_matrix_unit #(
  parameter int unsigned N = 16
);
  localparam int unsigned MAXR = 4096;
  localparam int unsigned MAXW = 2048;
  localparam int unsigned L    = 2 * N;
  localparam int unsigned G2   = 2 * N - 2;

  logic clk = 1'b0, rst_n = 1'b0;
  always #5 clk = ~clk;

  logic             x_valid = 1'b0, x_ready, x_new_w = 1'b0;
  logic [N*8-1:0]   x_data  = '0;
  logic             w_valid = 1'b0, w_ready;
  logic [N*8-1:0]   w_data  = '0;
  logic             y_valid, busy, w_stall, w_shift;
  logic [N*32-1:0]  y_data;

  matrix_unit #(.N(N)) dut (
    .clk_i(clk), .rst_ni(rst_n),
    .x_valid_i(x_valid), .x_ready_o(x_ready), .x_data_i(x_data), .x_new_w_i(x_new_w),
    .w_valid_i(w_valid), .w_ready_o(w_ready), .w_data_i(w_data),
    .y_valid_o(y_valid), .y_data_o(y_data),
    .busy_o(busy), .weight_stall_o(w_stall), .weight_shift_o(w_shift)
  );

  // ------------------------------------------------------------------------------------------
  // vectors
  // ------------------------------------------------------------------------------------------
  logic [N*8-1:0]  xmem [MAXR];
  logic [0:0]      xnew [MAXR];
  logic [N*8-1:0]  wmem [MAXW];
  logic [N*32-1:0] ymem [MAXR];
  int unsigned rows, words, blocks, metaN;
  string vec;
  int unsigned bubble_pct = 20, wgap_pct = 30, seed = 1;
  int unsigned errors = 0, checked = 0;
  longint unsigned stall_cycles = 0, shift_cycles = 0;

  task automatic fail(input string m);
    errors++;
    if (errors <= 20) $display("[ERROR] %s", m);
  endtask

  task automatic load_vectors();
    int fd, v;
    string key;
    fd = $fopen({vec, "/meta.txt"}, "r");
    if (fd == 0) $fatal(1, "cannot open %s/meta.txt", vec);
    while ($fscanf(fd, "%s %d", key, v) == 2) begin
      case (key)
        "N":      metaN  = v;
        "rows":   rows   = v;
        "words":  words  = v;
        "blocks": blocks = v;
        default: ;
      endcase
    end
    $fclose(fd);
    if (metaN != N) $fatal(1, "vector N=%0d but TB N=%0d", metaN, N);
    if (rows > MAXR || words > MAXW) $fatal(1, "vectors too large");
    $readmemh({vec, "/x.hex"},    xmem, 0, rows - 1);
    $readmemh({vec, "/xnew.hex"}, xnew, 0, rows - 1);
    $readmemh({vec, "/w.hex"},    wmem, 0, words - 1);
    $readmemh({vec, "/y.hex"},    ymem, 0, rows - 1);
  endtask

  // C4: independent reference (tile word t = W[N-1-t])
  task automatic check_golden();
    logic signed [7:0] W [N][N];
    int unsigned tile = 0, bad = 0;
    for (int unsigned r = 0; r < rows; r++) begin
      if (xnew[r]) begin
        for (int t = 0; t < N; t++)
          for (int c = 0; c < N; c++) W[N-1-t][c] = wmem[tile*N + t][c*8 +: 8];
        tile++;
      end
      for (int c = 0; c < N; c++) begin
        logic signed [31:0] acc = 0;
        for (int k = 0; k < N; k++) acc += 32'(signed'(xmem[r][k*8 +: 8])) * 32'(W[k][c]);
        if (acc !== signed'(ymem[r][c*32 +: 32])) bad++;
      end
    end
    if (tile * N != words) fail($sformatf("C4: %0d tiles referenced, %0d words supplied", tile, words));
    if (bad) fail($sformatf("C4: golden y.hex disagrees with in-TB reference in %0d values", bad));
    else $display("[C4] golden y.hex matches the independent in-TB reference (%0d rows)", rows);
  endtask

  // ------------------------------------------------------------------------------------------
  // one run: drive x (with bubbles) and w (with gaps), check every y in order and on time
  // abort_at > 0: assert reset after that many accepted rows and stop
  // ------------------------------------------------------------------------------------------
  longint unsigned cyc = 0;
  always @(posedge clk) cyc <= cyc + 1;

  task automatic run(input string name, input int unsigned abort_at = 0);
    int unsigned xi = 0, wi = 0, yi = 0, guard = 0, timeout = 0;
    longint unsigned acc_cyc [$];
    bit aborted = 0;
    while (yi < rows && !aborted) begin
      @(negedge clk);
      if (xi < rows && ($urandom_range(0, 99) >= bubble_pct)) begin
        x_valid = 1'b1; x_data = xmem[xi]; x_new_w = xnew[xi];
      end else begin
        x_valid = 1'b0; x_new_w = 1'b0;
      end
      if (wi < words && ($urandom_range(0, 99) >= wgap_pct)) begin
        w_valid = 1'b1; w_data = wmem[wi];
      end else w_valid = 1'b0;

      @(posedge clk);
      // C1
      if (!x_ready && !(x_valid && x_new_w)) fail($sformatf("%s C1: x_ready low on a non-new-weights cycle", name));
      // C2
      if (w_shift && guard != 0) fail($sformatf("%s C2: weight shift %0d cycles after a swap row", name, G2 - guard + 1));
      if (guard != 0) guard--;
      if (x_valid && x_ready && x_new_w) guard = G2;
      if (w_stall) stall_cycles++;
      if (w_shift) shift_cycles++;
      if (w_valid && w_ready) wi++;
      // C3
      if (y_valid) begin
        if (yi >= rows) fail($sformatf("%s C3: extra output row", name));
        else begin
          longint unsigned a = acc_cyc.pop_front();
          if (cyc - a != L) fail($sformatf("%s C3: row %0d latency %0d (expected %0d)", name, yi, cyc - a, L));
          if (y_data !== ymem[yi]) fail($sformatf("%s row %0d: got %h exp %h", name, yi, y_data, ymem[yi]));
          checked++;
          yi++;
        end
      end
      if (x_valid && x_ready) begin acc_cyc.push_back(cyc); xi++; end
      if (abort_at != 0 && xi >= abort_at) aborted = 1;
      if (++timeout > 50 * (rows + words) + 10000) begin fail($sformatf("%s: timeout xi=%0d wi=%0d yi=%0d", name, xi, wi, yi)); break; end
    end
    @(negedge clk); x_valid = 1'b0; x_new_w = 1'b0; w_valid = 1'b0;
    if (aborted) begin
      // mid-run reset: everything in flight is dropped
      rst_n = 1'b0; repeat (3) @(negedge clk); rst_n = 1'b1; repeat (2) @(negedge clk);
      $display("[RUN] %-12s aborted by reset after %0d rows (%0d outputs checked)", name, xi, yi);
    end else begin
      repeat (L + 2) @(posedge clk);
      if (y_valid || busy) fail($sformatf("%s C3: outputs/busy after the last row", name));
      if (wi != words) fail($sformatf("%s: %0d of %0d weight words consumed", name, wi, words));
      $display("[RUN] %-12s %0d rows, %0d words checked", name, rows, words);
    end
  endtask

  initial begin
    if (!$value$plusargs("VEC=%s", vec)) $fatal(1, "+VEC=<dir> required");
    void'($value$plusargs("BUBBLE_PCT=%d", bubble_pct));
    void'($value$plusargs("WGAP_PCT=%d", wgap_pct));
    void'($value$plusargs("SEED=%d", seed));
    void'($urandom(seed));
    load_vectors();
    $display("tb_matrix_unit: N=%0d vec=%s rows=%0d words=%0d blocks=%0d bubble=%0d%% wgap=%0d%% L=%0d",
             N, vec, rows, words, blocks, bubble_pct, wgap_pct, L);
    check_golden();
    repeat (3) @(negedge clk); rst_n = 1'b1; repeat (2) @(negedge clk);

    run("run1");
    run("run2");                       // back-to-back, no reset (L14)
    run("run3/reset", rows / 2 + 1);   // aborted by a mid-run reset
    run("run4");                       // clean after reset

    $display("tb_matrix_unit: checked=%0d rows, weight_stall=%0d cycles, weight_shift=%0d cycles, errors=%0d",
             checked, stall_cycles, shift_cycles, errors);
    if (errors == 0) $display("M3_MXU_PASS"); else $display("M3_MXU_FAIL errors=%0d", errors);
    $finish;
  end

endmodule
