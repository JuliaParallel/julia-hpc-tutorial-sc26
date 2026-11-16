## Requirements

Install IJulia and the multi-threaded kernel with

```
julia --project=. -e 'using Pkg; Pkg.instantiate(); using IJulia; IJulia.installkernel("Multithreaded Julia", "--project=@.", "--threads=auto")'
```

## Usage

Start Jupyterlab with

```
julia --project=. -e 'using IJulia; IJulia.jupyterlab(; dir=pwd())'
```
