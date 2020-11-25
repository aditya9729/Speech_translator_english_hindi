# Speech translator english to hindi
Uses Automatic Speech Recognition, Text to Speech voice synthesis and Neural Machine Translation to translate english to hindi

### Important notes: 
  * src directory has most of the .py files including ingestion, featurization and some model experiments
  * notebooks directory has all the experimentation notebooks
  * truncated_data has a zip folder please unzip and use, its a sample of the HindEnCorp parallel corpus.
  * static has index.css and app.js for styling
  * templates has the index.html page to be rendered
  
### Running the docker container
  * First and foremost please use the following command and record your voice to be translated. Your voice should be recorded as soon as it says recording..
  * It will be saved as a input.wav file in the static directory.
  
  * Command to get recording: 
     ```
     python ./src/create_audio_recording.py
     ```
  * Second build the speech translation service docker image with python 3.7.6 as thebase image - this installs all packages and gets model data.
  
  * Command to build docker image: 
    ``` 
    docker build -t speech-translation-service .
    ```
  * Finally deploy your container.
  
  * Command to deploy container: 
    ``` 
    docker run -p 5000:5000 -v "${pwd}"/app/data -it speech-translation-service
    ```
### Navigating the app
 * Once you run the container use 127.0.0.1:5000 to use the app on windows else let it remain the same. If there are problems please let us know.
 * The app should get you to a two audio controls. Upload the input.wav recorded file from the static folder and press the transcribe button.
 * You should then see english and hindi translation popping up and play the two audios - one is the original english version and the other is the hindi version

  
