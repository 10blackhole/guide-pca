using MultivariateStats, Statistics, Plots
using RDatasets: dataset

iris = dataset("datasets", "iris")
iris_matrix = Array(iris[:, 1:4])'

## PCA model:
M = fit(PCA, iris_matrix; pratio=1, maxoutdim=4)

iris_transformed = transform(M, iris_matrix)

h = plot(iris_transformed[1,:], iris_transformed[2,:], seriestype=:scatter, label="")
plot!(xlabel="PC1", ylabel="PC2", framestyle=:box)

for i=1:4; plot!([0, proj[i,1]], [0,proj[i,2]], arrow=true, label=names(iris)[i],
             legend=:bottomleft); end
display(h)
