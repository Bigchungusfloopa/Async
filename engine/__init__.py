"""Finance engine (P1). Pure python; no Streamlit."""
import warnings

# numpy 2.x on macOS (Accelerate BLAS) raises spurious floating-point flags from matmul even when every
# value is finite. Inputs here are validated (positive prices, finite weights), so this exact message is noise.
warnings.filterwarnings("ignore", message=r".*encountered in matmul", category=RuntimeWarning)
