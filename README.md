## cEBMF
Package implementing cEBMF method





## Optional R functionality (`mixsqp`)
When a factor has no covariate then the optimization is done via EM or via sequential quadratic optimization.

This optimization can be done extremly efficiently using the work of Kim et al. "fast algorithm for maximum likelihood estimation of mixture proportions using sequential quadratic programming."  the Journal of Computational and Graphical Statistics. 2020. For this option you must install, rpy2>= 3.0, R, the R package `mixsqp`, and the optional Python dependency:

```bash
pip cEBMF[r]
```
If R2py is not installed then the optimization will be done using an EM algorithm that can be slow.

```R
install.packages("remotes")
remotes::install_github("stephenslab/mixsqp")
``` 