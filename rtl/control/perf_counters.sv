// TPU v1 Table 3 style performance counters (docs/m4_compute_core.md sec 7). Saturating 32-bit.
//   [0] total  [1] array_active  [2] weight_stall  [3] weight_shift  [4] non_matrix
//   [5] raw_stall (SYNC)  [6] act_active  [7] instr_count
module perf_counters (
  input  logic        clk_i,
  input  logic        rst_ni,
  input  logic        clr_i,
  input  logic [7:0]  inc_i,
  output logic [31:0] cnt_o [8]
);
  for (genvar k = 0; k < 8; k++) begin : gen_cnt
    always_ff @(posedge clk_i or negedge rst_ni) begin
      if (!rst_ni)                         cnt_o[k] <= '0;
      else if (clr_i)                      cnt_o[k] <= '0;
      else if (inc_i[k] && ~&cnt_o[k])     cnt_o[k] <= cnt_o[k] + 32'd1;
    end
  end
endmodule
