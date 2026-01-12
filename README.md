# One Shot Method for Computing Generalized Winding Numbers

[Project Page](https://martenscedric.github.io/academic-page/publications/1s_wn.html)

# Citing

```
@article{Martens2025WindingNumberOneShot,
  title = {One-Shot Method for Computing Generalized Winding Numbers},
  author = {Martens, Cedric and Bessmeltsev, Mikhail},
  journal = {Computer Graphics Forum},
  doi = {10.1111/cgf.70194},
  volume = {44},
  number = {5},
  year = {2025},
}
```

# Instructions

## Building
```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
```

## Running Examples

### Mesh

```
cd build/3d/
./one_shot_wn_3d  -n camel -m ../../inputs/camelhead.obj -q ../../inputs/camelhead_500_300.points
cd ../../
matlab -nodesktop -nosplash -nojvm -softwareopengl -batch "name='camel'; run('3d/visualize_results.m');"
```

