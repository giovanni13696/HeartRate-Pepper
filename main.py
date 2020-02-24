from __future__ import with_statement
from __future__ import division
from __future__ import absolute_import
import cv2
import numpy as np
import time
import sys
import argparse
import qi
import os
from imutils import face_utils
from face_utilities import Face_utilities
from signal_processing import Signal_processing
from io import open


class HumanGreeter(object):

    def __init__(self, app):
        """
            Inizializzazione.
        """
        super(HumanGreeter, self).__init__()
        app.start()
        session = app.session

        # Servizi di Pepper necessari per l'algoritmo
        self.memory = session.service("ALMemory")
        self.tts = session.service("ALTextToSpeech")
        self.asr_service = session.service("ALSpeechRecognition")
        self.face_detection = session.service("ALFaceDetection")
        self.audioRecorder = session.service("ALAudioRecorder")
        self.posture = session.service("ALRobotPosture")
        self.videoRecorder = session.service("ALVideoRecorder")
        self.face_detection.subscribe("HumanGreeter")
        self.got_face = False
        self.asr_service.pause(True)
        self.asr_service.setParameter("NbHypotheses", 1)
        self.asr_service.setLanguage("English")
        vocabulary = ["battito"]
        self.asr_service.setVocabulary(vocabulary, True)
        self.asr_service.pause(False)
        self.face = False

    def on_human_tracked(self, value):
        if not value:  # Valore vuoto quando la faccia scompare
            self.got_face = False
        elif not self.got_face and self.face == False:  # Quando una faccia viene riconosciuta...
            self.got_face = True

    # Conversazione iniziale con Pepper
    def conversation(self):
        self.tts.say("Hello, what do you want to do?")
        time.sleep(2)

        self.asr_service.subscribe("action")
        time.sleep(2)

        action = self.memory.getData("WordRecognized")
        self.asr_service.unsubscribe("action")

        # Se gli chiediamo di rilevare il battito cardiaco...
        if action[0] == "<...> battito <...>":
            self.tts.say("Okay, I'm going to capture a 40 seconds video, then I will give you your heart rate")
            self.tts.say("Please, face the camera and don't move!")
            self.run()

    # Algoritmo di rilevamento del battito
    def run(self):

        i = 0
        last_rects = None
        last_shape = None
        last_age = None
        last_gender = None
        face_detect_on = False
        age_gender_on = False
        times = []
        data_buffer = []
        fft_of_interest = []
        freqs_of_interest = []
        valori = []
        bpm = 0

        # Cattura ed apertura del video
        path = self.record()
        cap = cv2.VideoCapture(path)

        # Se cap e' nullo vuol dire che il video non e' stato aperto correttamente
        if cap is None:
            print "Errore nell'apertura del video"
            return

        fu = Face_utilities()
        sp = Signal_processing()

        t = time.time()
        BUFFER_SIZE = 100

        fps = 0
        video_fps = cap.get(cv2.CAP_PROP_FPS)

        '''
            Loop infinito. Ogni ciclo equivale alla lettura di un frame del video. 
        '''
        while True:

            t0 = time.time()

            if i % 1 == 0:
                face_detect_on = True
                if i % 10 == 0:
                    age_gender_on = True
                else:
                    age_gender_on = False
            else:
                face_detect_on = False

            # Lettura del frame
            ret, frame = cap.read()

            # Se frame e' nullo vuol dire che il video e' finito. Stop al programma.
            if frame is None:
                print "Video terminato - Nessun frame disponibile"
                cv2.destroyAllWindows()
                break

            # Face detection con i 68 landmarks.
            ret_process = fu.no_age_gender_face_process(frame, u"68")

            # Se ret_process e' nullo vuol dire che i landmarks non sono stati applicati correttamente
            # quindi nessun volto e' stato rilevato.
            # Controllo inutile se non si guarda il pc durante l'esecuzione dell'algoritmo.
            if ret_process is None:
                cv2.putText(frame, u"Nessun volto rilevato", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow(u"Frequenza Cardiaca", frame)
                print time.time() - t0

                if cv2.waitKey(1) & 0xFF == ord(u'q'):
                    cv2.destroyAllWindows()
                    break
                continue

            rects, face, shape, aligned_face, aligned_shape = ret_process

            (x, y, w, h) = face_utils.rect_to_bb(rects[0])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Disegno dei rettangoli su guancia destra e sinistra
            if len(aligned_shape) == 68:
                cv2.rectangle(aligned_face, (aligned_shape[54][0], aligned_shape[29][1]),
                              (aligned_shape[12][0], aligned_shape[33][1]), (0, 255, 0), 0)
                cv2.rectangle(aligned_face, (aligned_shape[4][0], aligned_shape[29][1]),
                              (aligned_shape[48][0], aligned_shape[33][1]), (0, 255, 0), 0)
            else:
                cv2.rectangle(aligned_face, (aligned_shape[0][0], int((aligned_shape[4][1] + aligned_shape[2][1]) / 2)),
                              (aligned_shape[1][0], aligned_shape[4][1]), (0, 255, 0), 0)

                cv2.rectangle(aligned_face, (aligned_shape[2][0], int((aligned_shape[4][1] + aligned_shape[2][1]) / 2)),
                              (aligned_shape[3][0], aligned_shape[4][1]), (0, 255, 0), 0)

            for (x, y) in aligned_shape:
                cv2.circle(aligned_face, (x, y), 1, (0, 0, 255), -1)

            # Estrazione delle caratteristiche
            ROIs = fu.ROI_extraction(aligned_face, aligned_shape)

            # Estrazione del valore di verde dalle ROI
            green_val = sp.extract_color(ROIs)

            # Inserimento del valore di verde in un data buffer
            data_buffer.append(green_val)

            times.append((1.0 / video_fps) * i)

            L = len(data_buffer)

            if L > BUFFER_SIZE:
                data_buffer = data_buffer[-BUFFER_SIZE:]
                times = times[-BUFFER_SIZE:]
                # bpms = bpms[-BUFFER_SIZE//2:]
                L = BUFFER_SIZE

            # Non appena il buffer e' stato riempito con almeno 100 valori si inizia a stampare il battito cardiaco
            # Quindi dopo il passaggio di 100 frames.
            if L == 100:
                fps = float(L) / (times[-1] - times[0])
                cv2.putText(frame, u"fps: {0:.2f}".format(fps), (30, int(frame.shape[0] * 0.95)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 255, 255), 2)

                detrended_data = sp.signal_detrending(data_buffer)
                interpolated_data = sp.interpolation(detrended_data, times)

                normalized_data = sp.normalization(interpolated_data)

                fft_of_interest, freqs_of_interest = sp.fft(normalized_data, fps)

                max_arg = np.argmax(fft_of_interest)
                bpm = freqs_of_interest[max_arg]
                cv2.putText(frame, u"HR: {0:.2f}".format(bpm), (int(frame.shape[1] * 0.8), int(frame.shape[0] * 0.95)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                filtered_data = sp.butter_bandpass_filter(interpolated_data, (bpm - 20) / 60, (bpm + 20) / 60, fps,
                                                          order=3)

            # Apertura (o creazione) del file result.txt e scrittura del valore del battito
            with open(u"result.txt", mode=u"a+") as f:
                f.write(u"time: {0:.4f} ".format(times[-1]) + u", HR: {0:.2f} ".format(bpm) + u"\n")

                # Se il battito e' significativo quindi maggiore di 70, si inserisce il valore in un array che servira'
                # per calcolare la media del battito finale
                if bpm > 65:
                    valori.append(bpm)

            i = i + 1

            # Allo scorrere dei frame viene stampato il numero del frame corrente
            print u"Frame numero " + unicode(i) + u" : " + unicode(time.time() - t0)

            if cv2.waitKey(1) & 0xFF == ord(u'q'):
                cv2.destroyAllWindows()
                break

        cap.release()
        cv2.destroyAllWindows()

        # Pepper dice la media dei valori del battito cardiaco
        self.tts.say("Your heart rate is " + format(np.mean(valori)))

        print u"Tempo totale impiegato: " + unicode(time.time() - t)

    def record(self):

        # Path e nome del video registrato da Pepper
        # e path della destinazione locale del file
        folderPath = "/home/nao/recordings/cameras"
        localPath = "/Users/GiovanniMusacchio/Desktop"
        fileName = "face.avi"
        count = 0

        self.videoRecorder.setFrameRate(15)
        self.videoRecorder.setResolution(2)

        # Start della registrazione video
        self.videoRecorder.startRecording(folderPath, fileName)
        print "Registrazione iniziata"

        # Pepper sta fermo per circa 40 secondi
        while count < 40:
            self.posture.goToPosture("StandInit", 1.0)
            count = count+1

        # Stop della registrazione video
        videoInfo = self.videoRecorder.stopRecording()

        # Inserimento in memoria delle informazioni del video
        self.memory.insertData("video", videoInfo)
        print "Registrazione terminata"

        self.tts.say("Video successfully captured")
        self.tts.say("Please, wait 20 seconds!")

        print "------------------------"
        print "Video salvato in: ", videoInfo[1]
        print "Frame totali: ", videoInfo[0]
        print "------------------------"

        '''
            Di seguito viene eseguito il trasferimento del file da remoto a locale
            mediante terminale. Viene usato il comando scp su macOS quindi in 
            ambiente Unix, per ambiente Windows provare pscp per scaricare tramite
            PuTTY (ssh) 
            
            "scp FROM TO"
            sshpass -p serve per passare come parametro la password per accedere a nao@pepper.local      
         '''

        os.system('sshpass -p "biplab" scp nao@pepper.local:/home/nao/recordings/cameras/face.avi ~/Desktop')

        # Ritorna il path completo locale con il nome del file
        path = localPath + "/face.avi"
        return path


if __name__ == u"__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="pepper.local",
                        help="Robot IP address. On robot or Local Naoqi: use '127.0.0.1'.")
    parser.add_argument("--port", type=int, default=9559,
                        help="Naoqi port number")
    args = parser.parse_args()
    try:
        # Initialize qi framework.
        connection_url = "tcp://" + args.ip + ":" + str(args.port)
        app = qi.Application(["HumanGreeter", "--qi-url=" + connection_url])
    except RuntimeError:
        print("Can't connect to Naoqi at ip \"" + args.ip + "\" on port "
              + str(args.port) + ".\n"
                                 "Please check your script arguments. Run with -h option for help.")
        sys.exit(1)

    human_greeter = HumanGreeter(app)
    human_greeter.conversation()
