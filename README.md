# AllSkyAI-RPi

## How to install

```bash
git clone https://github.com/TheCrowsNestAstro/AllSkyAI-RPi.git
cd AllSkyAI-RPi
python3 -m venv venv
source ./venv/bin/activate
python3 -m pip install --upgrade pip

wget https://dl.google.com/coral/python/tflite_runtime-1.14.0-cp37-cp37m-linux_armv7l.whl
pip3 install tflite_runtime-1.14.0-cp37-cp37m-linux_armv7l.whl

pip3 install -r requirements.txt
```


Modiy the example.env

```bash
mv example.env .env
nano .env
```
Download your trained models from https://www.allskyai.com and place the `*.tflite` and `labels_map*.txt` file in `/models` directory
Add the file names to the `.env` file and your AllSky URL

`COLOR_MODE` can be either `rgb` or `grayscale` depending on your camera

## Start the server
Simply start the server by

````bash
python3 waitress_server.py
````

Open a webbrowser and point to `<ip/server>:3010`

Wait for the result:
````json
{"classification": "light_clouds", "confidence": 95.853, "utc": 1684667437, "inference": 0.661}
````
## Copy files from Windows to RPi
Example of how to copy the model and text file over to your RPi
```bash
scp allskyai_2023-05-07-21-38-44.tflite pi@192.168.1.110:/home/pi/AllSkyAI-RPi/models
```
