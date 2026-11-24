# No need to touch this file, it's just used to set up the notebooks.

using Pkg
Pkg.activate(joinpath(@__DIR__))
Pkg.add(["BenchmarkTools", "ProgressMeter", "Plots", "ColorTypes", "KernelAbstractions", "PythonCall", "JSON"])

gpu_model = String(DEV)
if gpu_model == "NVIDIA"
    Pkg.add(["CUDA"])
elseif gpu_model == "AMD"
    Pkg.add(["AMDGPU"])
elseif gpu_model == "Metal"
    Pkg.add(["Metal"])
end

# Pkg.develop(path="../../../../KernelTuner.jl")
Pkg.add(["KernelTuner"])

function is_system_dark_mode()
    if Sys.isapple()
        # Checks macOS global preferences
        return success(`defaults read -g AppleInterfaceStyle`)
    elseif Sys.iswindows()
        # Checks Windows registry key for Dark Mode
        try
            reg_val = read(global_theme_cmd, String)
            return occursin("0x1", read(`reg query "HKCU\Software\Microsoft\Windows\CurrentVersion\Themes\Personalize" /v AppsUseLightTheme`, String)) == false
        catch
            return false
        end
    else
        return false # Fallback for Linux or unexpected environments
    end
end

# Check if the system is in dark mode if the user has not set the `dark_mode` variable themselves
if !isdefined(Main, :dark_mode_override)
    dark_mode = is_system_dark_mode()
else
    dark_mode = Main.dark_mode_override
end

# Load required packages
using JSON
using PythonCall
using Statistics
using ProgressMeter
using BenchmarkTools
using Plots
using ColorTypes
using KernelAbstractions
using KernelTuner

# Set the default theme based on system dark mode preference
if dark_mode
    theme(:dark)
else
    theme(:default)
end

# Define a lightweight text wrapper class to get live stdout to the cell
sys = pyimport("sys")
@pyexec """
class JuliaTextStdout:
    def __init__(self, jl_print):
        self.jl_print = jl_print
    def write(self, string):
        if string:
            self.jl_print(string)
        return len(string)
    def flush(self):
        pass
""" => JuliaTextStdout
sys.stdout = JuliaTextStdout(print)

# Define a function to get the source of a cell by tag
function get_cell_by_tag(tag_name)
    notebook = JSON.parsefile("kerneltuner.ipynb")
    cell_idx = findfirst(notebook["cells"]) do cell
        tags = get(get(cell, "metadata", Dict()), "tags", [])
        return tag_name in tags
    end
    if isnothing(cell_idx)
        error("No cell found with tag: $tag_name")
    end
    source = notebook["cells"][cell_idx]["source"]
    return join(source)
end

# The plotting helpers (`plot_tuning_heatmap`, `with_live_tuning_plot`, `plot_time_energy`, `pareto_front`, ...)
# are part of KernelTuner.jl, and become available because Plots is loaded above.
