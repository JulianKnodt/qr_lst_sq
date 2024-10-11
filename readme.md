# Rust QR Least Squares

```
A = QR
QRx = b
Rx = Q^Tb
```

A can be any of the following:

1. Compile time fixed rows and columns with #Rows >= #Columns
  - `mgs_qr(A)` then `qr_solve`
2. Compile time fixed columns with Dynamic #Rows >= #Columns
  - `dyn_mgs_qr(A)` then `dyn_qr_solve`
  - Uses a single output buffer `q`, will not allocate if there is sufficient space.
3. Compile time fixed columns with Rows < #Columns
  - `mgs_qr(transpose(A))` then `qr_solve_underdetermined`



Standalone Rust QR Least Squares Modified Gram Schmidt solver.
Thanks to Nicolas Boumal's notes for algorithm source.

See [this link](https://www.nicolasboumal.net/#teaching), numerical methods (MAT321) lecture
notes.


Feel free to file issues and use as you will.

![Pikachu](https://blog.archive.org/wp-content/uploads/2016/10/underconstruction.gif)
