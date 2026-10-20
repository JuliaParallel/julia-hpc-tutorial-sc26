include("main.jl")
include("solution.jl")

num_bands = 60
num_voxel = 4096
size_voxel = 64*64

img_h, filter_h = Ex_spectral.init(
    num_bands, num_voxel, size_voxel; dtype = JACC.default_float())

@time Ex_spectral.spectral!(num_voxel, size_voxel, img_h, filter_h, num_bands)
@time Ex_spectral.spectral!(num_voxel, size_voxel, img_h, filter_h, num_bands)

println()

image = JACC.to_device(img_h)
filter = JACC.to_device(filter_h)

@time spectral_jacc!(num_voxel, size_voxel, image, filter, num_bands)
@time spectral_jacc!(num_voxel, size_voxel, image, filter, num_bands)

println()

@time spectral_jacc_shared!(num_voxel, size_voxel, image, filter, num_bands)
@time spectral_jacc_shared!(num_voxel, size_voxel, image, filter, num_bands)
