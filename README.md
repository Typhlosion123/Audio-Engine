# Audio-Engine

## How to use the Typhlosion123 Audio Ray Tracer

After cloning the repo, complete the following commands:

```bash
mkdir build
cd build
cmake ..
make 
```

This will allow you to use the CUDA ray tracer. We will not move onto the python step. 

### Python steps

First, make sure to install the neccessary packages. Go to a new terminal and run the following commands:

```bash
cd python
pip install -r requirements.txt
```

There are two ways to create your walls/spheres. 

##### The easy way

Go to a new terminal and run the following commands:

```bash
streamlit run builder.py
```

This is an online builder that allows you to interactively create your audio environment. When you are done, save the sketch. 

#### Harder way

If you are the goat (which you probably are <3), then go to ```scene_builder.py``` and try making your own scene from there. Once you are down editing scene_builder, run the code to save the ```scene.bin``` to your build folder. 

### Everything else

Now that everything is ready, run:

```bash
./AcousticSim
```

This will create a paths.csv in your build folder. Now go to ```graph_rays.py``` and run that script. This will create a file called ```audio_sim.html``` in your ```python/output``` folder. This can be used in a browser to see your creation. 

Lastly, head over to ```auralizer.py``` and run the script there. After a little bit, you will get ```audio_sim.wav```. Play this to see how audio would travel in that room with that spec with the power of ray tracing :)


### Final thoughts

If you are too lazy to do allat, head on over to [my website](https://www.chrisyxu.com) and go to the projects. There you can find a demo just for you!