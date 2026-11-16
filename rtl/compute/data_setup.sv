// Systolic data setup: lane k (with valid and swap) is delayed by k cycles so that one input row
// enters the array as a diagonal wavefront (TPU v1 sec 2). Row 0 is not delayed.

module data_setup #(
  parameter int unsigned N = 16
) (
  input  logic                clk_i,
  input  logic                rst_ni,
  input  logic [N-1:0][7:0]   data_i,
  input  logic                valid_i,
  input  logic                swap_i,
  output logic [N-1:0][7:0]   data_o,
  output logic [N-1:0]        valid_o,
  output logic [N-1:0]        swap_o
);

  assign data_o[0]  = data_i[0];
  assign valid_o[0] = valid_i;
  assign swap_o[0]  = swap_i && valid_i;

  for (genvar k = 1; k < N; k++) begin : gen_lane
    logic [k-1:0][7:0] d_q;
    logic [k-1:0]      v_q, s_q;

    always_ff @(posedge clk_i or negedge rst_ni) begin
      if (!rst_ni) begin
        v_q <= '0;
        s_q <= '0;
      end else begin
        v_q[0] <= valid_i;
        s_q[0] <= swap_i && valid_i;
        for (int i = 1; i < k; i++) begin
          v_q[i] <= v_q[i-1];
          s_q[i] <= s_q[i-1];
        end
      end
    end

    always_ff @(posedge clk_i) begin
      d_q[0] <= data_i[k];
      for (int i = 1; i < k; i++) d_q[i] <= d_q[i-1];
    end

    assign data_o[k]  = d_q[k-1];
    assign valid_o[k] = v_q[k-1];
    assign swap_o[k]  = s_q[k-1];
  end

endmodule
