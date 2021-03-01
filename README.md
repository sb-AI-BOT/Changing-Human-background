# Changing-Human-Background

<h2 align="center"> Is a Green or Blue Screen Really Necessary for Real-Time Portrait Matting?</h2>

| From | To |
| --- | --- |
|![iron_org](https://user-images.githubusercontent.com/67555058/109460983-870ecd80-7a87-11eb-8be1-24dc16568ac4.gif)| ![iron](https://user-images.githubusercontent.com/67555058/109461012-92fa8f80-7a87-11eb-8885-8149af10bd89.gif)| 


### Git Clone
```sh
git clone https://github.com/sb-AI-BOT/Changing-Human-background.git
```


## Installation
### Requirements
* `pip install -r requirements.txt`

Note: See [requirements.txt](requirements.txt) for more details.


### Pre-Trained Model
Download the [**human.pth**](https://drive.google.com/file/d/1--R6edScQB05EebPu13OTCVmDYoElgPV/view?usp=sharing) model and put it in directory: ```./model/```.


### Run the process
```sh
python run_bg.py --src_img --bg_img
```
## More Results
| From | To |
| --- | --- |
| ![elonmusk](https://user-images.githubusercontent.com/67555058/109463967-eb339080-7a8b-11eb-9403-6ebe3e935ba2.jpg)| ![elonmusk](https://user-images.githubusercontent.com/67555058/109463981-f090db00-7a8b-11eb-85ff-f6b50a63a78c.png)|

