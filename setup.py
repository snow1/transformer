from dtu import setup

# see 'module available' on server for newest python version
setup(
    f"https://github.com/snow1/transformer.git",
    python="3.9.6",
    packages=["torch", "torchvision", "matplotlib", "einops", "pandas", "scipy", "torchsummary"]
)
