import numpy as np
import cv2 as cv
from cv2 import aruco as aruco
import glob

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((5*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:5].T.reshape(-1,2)

# arrays para guardar pontos
objpoints = [] # pontos 3d no mundo real
imgpoints = [] # pontos 2d na imagem

# importar imagens para a calibração
images = glob.glob('db\*.jpg')

# fazer para cada imagem
for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # encontrar o centro do tabuleiro
    ret, corners = cv.findChessboardCorners(gray, (7,5), None)

    # se encontrar o centro adiciona os pontos
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)

        # dezenha e mostra os cantos
        cv.drawChessboardCorners(img, (7,5), corners2, ret)
        cv.imshow('img', img)
        # cv.imwrite('results\grid4.png', img)
        cv.waitKey(500)
        # print(objpoints[0].shape)
cv.destroyAllWindows()

# vars de calibração
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# definir dicionario do aruco
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)  # You can choose a different dictionary if needed
parameters = aruco.DetectorParameters()

# iniciar a camera
# index da camera
camera = cv.VideoCapture(0)  
# resolução da camera
camera.set(cv.CAP_PROP_FRAME_WIDTH, 1280) 
camera.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

# tamanho do marker e tamanho do cubo
marker_size = 0.05
cube_size = 0.1
pyramid_size = 0.1

# para cada frame da camera
while True:
    ret, frame = camera.read()  # captura o frame
    
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)  # converte o frame para grayscale
    
    # deteta o marcador
    corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    
    # quando encontra o marcador
    if ids is not None:
        
        
        # para cada marcador
        for i in range(len(ids)):
            # Get the ID of the detected marker
            marker_id = ids[i][0]

            # dezenha o limite do marcador
            aruco.drawDetectedMarkers(frame, corners, ids)

            # estima a posicao do marcador
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_size, mtx, dist)

            # obtem os vetores de rotação e translação 
            rvec, tvec = rvecs[i], tvecs[i]

            if marker_id == 0:
                # definir os pontos 3d no espaço em relação ao marcador
                points, _ = cv.projectPoints(cube_size * np.float32([[-0.25, -0.25, 0],
                                                                    [0.25, -0.25, 0],
                                                                    [0.25, 0.25, 0],
                                                                    [-0.25, 0.25, 0],
                                                                    [-0.25, -0.25, 0.5],
                                                                    [0.25, -0.25, 0.5],
                                                                    [0.25, 0.25, 0.5],
                                                                    [-0.25, 0.25, 0.5]]),
                                            rvec, tvec, mtx, dist)
                
                points = np.int32(points).reshape(-1, 2)  # converte os pontos para array 
        
                # definir as faces do cubo utilizando os pontos definidos
                cube_faces = [(points[0], points[1], points[2], points[3]),  # Top face
                            (points[4], points[5], points[6], points[7]),  # Bottom face
                            (points[0], points[1], points[5], points[4]),  # Side face 1
                            (points[1], points[2], points[6], points[5]),  # Side face 2
                            (points[2], points[3], points[7], points[6]),  # Side face 3
                            (points[3], points[0], points[4], points[7])]  # Side face 4
                
                # dezenha as faces do cubo 
                for face in cube_faces:
                    cv.drawContours(frame, [np.array(face)], -1, (0, 0, 255), cv.FILLED)

                # dezenha as arestas do cubo a azul
                cv.drawContours(frame, [points[:4]], -1, (255, 0, 0), 2)  # arestas da face de cima
                for i in range(4):
                    cv.line(frame, tuple(points[i]), tuple(points[i + 4]), (255, 0, 0), 2)  # arestas verticais do cubo
                cv.drawContours(frame, [points[4:]], -1, (255, 0, 0), 2)  # arestas da face de baixo

            elif marker_id == 1:
                # definir os pontos 3d no espaço em relação ao marcador
                points, _ = cv.projectPoints(pyramid_size * np.float32([[0, 0, 0.5], # ponto do vertice - 0
                                                                        [0.25, 0.25, 0], # ponto superior esquerdo - 1
                                                                        [0.25, -0.25, 0], # ponto superior direito - 2
                                                                        [-0.25, -0.25, 0], # ponto inferior direito - 3
                                                                        [-0.25, 0.25, 0]]), # ponto inferior esquerdo - 4
                                            rvec, tvec, mtx, dist)
                
                points = np.int32(points).reshape(-1, 2)  # Convert points to numpy array
                
                # faces das piramides
                pyramid_faces = [(points[0], points[1], points[2]),  # face distante
                                (points[0], points[1], points[4]),  # face esquerda
                                (points[0], points[4], points[3]),  # face frente
                                (points[0], points[3], points[2]),  # face direita
                                (points[1], points[3], points[4]),  # base 1
                                (points[1], points[3], points[2])]  # base 2

                for face in pyramid_faces:
                    cv.drawContours(frame, [np.array(face)], -1, (0, 255, 0), cv.FILLED)
        
                # dezenha as arestas do cubo a azul
                cv.line(frame, tuple(points[0]), tuple(points[1]), (255, 0, 0), 2)  # aresta vertical
                cv.line(frame, tuple(points[0]), tuple(points[2]), (255, 0, 0), 2)  # aresta vertical
                cv.line(frame, tuple(points[0]), tuple(points[3]), (255, 0, 0), 2)  # aresta vertical
                cv.line(frame, tuple(points[0]), tuple(points[4]), (255, 0, 0), 2)  # aresta vertical
                cv.line(frame, tuple(points[1]), tuple(points[2]), (255, 0, 0), 2)  # aresta da base
                cv.line(frame, tuple(points[2]), tuple(points[3]), (255, 0, 0), 2)  # aresta da base
                cv.line(frame, tuple(points[3]), tuple(points[4]), (255, 0, 0), 2)  # aresta da base
                cv.line(frame, tuple(points[4]), tuple(points[1]), (255, 0, 0), 2)  # aresta da base

    cv.imshow('Objeto a detetar', frame)  # mostrar o frame
    
    if cv.waitKey(1) & 0xFF == ord('q'):  # sair da janela
        break

camera.release()  # libertar o acesso á camera
cv.destroyAllWindows()  # quit