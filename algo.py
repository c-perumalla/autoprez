'''             
                                                                        ##########
                                                                        ## ALGO ##
                                                                        ##########

                                    song data structure (stored on disk)
                                            |
                                            |
                                            V
                                 -----------------------
Input from live sound           |         ALGO          |
(mic)                  ------>  |     (Whisper Live)    | ------> [ correct(True/False, conf,0-1), transition(True/False, conf,0-1), %complete  ]
                                |                       |         (saves to disk)                                                   |
                                 -----------------------                |                                                           |
                                                                        |                                                           |
                                                                        |                                                           |
                                                                        |                                                           |
                                                                        V                                                           V
                                                                     -----------                                         -----------------------
                                                                    |           |                                       |                       |
                                                                    |  DISPLAY  | ----> "Song String" --------------->  |   Measure and report  |     
                                                                    |           |                                       |    accuracy metrics   |
                                                                     -----------                                         -----------------------

'''
# imports 
from whisper_live.client import TranscriptionClient
import sys

def initiate_model():
    client = TranscriptionClient(
    "localhost",
    9090,
    lang="en",
    translate=False,
    model="small",
    use_vad=False,
    )

def main():
    # load args
    args = sys.argv[1:]
    song_title = args[1]
    # load song stucture from disk based on args
    song_structre = load_song_title(song_title)
    
    # Initial while loop to 'listen' to the mic
    # initiate whisper model
    initiate_model()
    

    # log and print output 
    # Measure and report metrics
    pass


if __name__ == '__main__':
    main()