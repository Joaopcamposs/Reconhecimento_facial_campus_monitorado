from pathlib import Path
from time import sleep

from cv2 import VideoCapture
from fastapi import APIRouter, BackgroundTasks, Depends, File, Query, UploadFile
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from starlette.responses import StreamingResponse

from src.entities.schemas import CreateAndUpdateCamera, CreateAndUpdatePerson
from src.infra.config import (
    BASE_DIR,
    PICTURES_DIR,
    classifier_exists,
    get_webcam_capture,
)
from src.infra.database import get_db
from src.repositories.camera_repository import (
    create_camera,
    get_all_cameras,
    get_camera_by_id,
    remove_camera,
    update_camera,
)
from src.repositories.controller_repository import set_capture_flag
from src.repositories.person_repository import (
    create_person,
    get_all_persons,
    get_person_by_id,
    remove_person,
    update_person,
)
from src.services.facial_recognition import (
    stream_facial_recognition,
    stream_recognition_only,
)
from src.services.pictures_capture import (
    get_capture_state,
    reset_capture_state,
    start_capture_session,
    stream_pictures_capture,
    stream_pictures_capture_auto,
    stream_video_only,
    trigger_capture,
)
from src.services.training import trainLBPH
from src.services.video_analysis import analyze_video_file, analyze_video_file_sync

# Directory for uploaded videos
VIDEOS_DIR: Path = BASE_DIR / "videos"
VIDEOS_DIR.mkdir(exist_ok=True)

# Directory for templates
TEMPLATES_DIR: Path = BASE_DIR / "templates"

app = APIRouter()


# API endpoint to check system status
@app.get("/status")
def verificar_status():
    """Check system status and available resources."""
    # Check webcam availability
    webcam_available = False
    try:
        cap = get_webcam_capture()
        webcam_available = cap.isOpened()
        cap.release()
    except Exception:
        pass

    # Check pictures directory
    pictures_count = len(list(PICTURES_DIR.glob("person.*.*.jpg")))

    return {
        "status": "online",
        "webcam_available": webcam_available,
        "classifier_trained": classifier_exists(),
        "pictures_count": pictures_count,
        "pictures_directory": str(PICTURES_DIR),
    }


# API endpoint to get info of a particular camera
@app.get("/camera/{camera_id}")
def pegar_info_camera(camera_id: int, session: Session = Depends(get_db)):
    try:
        camera_info = get_camera_by_id(session, camera_id)
        return camera_info
    except Exception as e:
        raise e


# API endpoint to update a existing camera info
@app.put("/camera/{camera_id}")
def atualizar_info_camera(
    camera_id: int,
    new_info: CreateAndUpdateCamera,
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_db),
):
    try:
        background_tasks.add_task(update_camera, session, camera_id, new_info)
        return 200, "Requisição recebida"
    except Exception as e:
        raise e


# API endpoint to get the list of cameras
@app.get("/cameras")
def listar_cameras(session: Session = Depends(get_db)):
    cameras = get_all_cameras(session=session)

    return cameras


# API endpoint to add a camera to the database
@app.post("/camera")
def cadastrar_camera(
    background_tasks: BackgroundTasks,
    new_camera: CreateAndUpdateCamera,
    session: Session = Depends(get_db),
):
    try:
        background_tasks.add_task(create_camera, session, new_camera)
        return 200, "Requisição recebida"
    except Exception as e:
        raise e


# API endpoint to delete a camera from the database
@app.delete("/camera/{camera_id}")
def deletar_camera(
    background_tasks: BackgroundTasks,
    camera_id: int,
    session: Session = Depends(get_db),
):
    try:
        background_tasks.add_task(remove_camera, session, camera_id)
        return 200, "Requisição recebida"
    except Exception as e:
        raise e


# API endpoint to get info of a particular pessoa
@app.get("/pessoa/{person_id}")
def pegar_info_pessoa(person_id: int, session: Session = Depends(get_db)):
    try:
        person_info = get_person_by_id(session, person_id)
        return person_info
    except Exception as e:
        raise e


# API endpoint to update a existing pessoa info
@app.put("/pessoa/{person_id}")
def atualizar_info_pessoa(
    person_id: int,
    new_info: CreateAndUpdatePerson,
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_db),
):
    try:
        background_tasks.add_task(update_person, session, person_id, new_info)
        return 200, "Requisição recebida"
    except Exception as e:
        raise e


# API endpoint to get the list of pessoas
@app.get("/pessoas")
def listar_pessoas(session: Session = Depends(get_db)):
    persons = get_all_persons(session=session)

    return persons


# API endpoint to add a pessoa to the database
@app.post("/pessoa")
def cadastrar_pessoa(
    background_tasks: BackgroundTasks,
    new_person: CreateAndUpdatePerson,
    session: Session = Depends(get_db),
):
    try:
        background_tasks.add_task(create_person, session, new_person)
        return 200, "Requisição recebida"
    except Exception as e:
        raise e


# API endpoint to delete a car info from the data base
@app.delete("/pessoa/{person_id}")
def deletar_pessoa(
    background_tasks: BackgroundTasks,
    person_id: int,
    session: Session = Depends(get_db),
):
    try:
        background_tasks.add_task(remove_person, session, person_id)
        return 200, "Requisição recebida"
    except Exception as e:
        raise e


# API endpoint to train a new file of facial recognition
@app.get("/treinamento")
def treinar_reconhecimento():
    try:
        success, message = trainLBPH()
        if success:
            return {"status": "success", "message": message}
        else:
            return {"status": "error", "message": message}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# API endpoint to facial recognition stream
@app.get("/video/{camera_id}")
def reconhecimento_facial(camera_id: int, session: Session = Depends(get_db)):
    return StreamingResponse(
        stream_facial_recognition(session=session, id_camera=camera_id),
        media_type="multipart/x-mixed-replace;boundary=frame",
    )


# API endpoint to stream and catch pictures
@app.get("/fotos/{camera_id}&{nome_pessoa}")
def capturar_fotos(
    camera_id: int, nome_pessoa: str, session: Session = Depends(get_db)
):
    try:
        return StreamingResponse(
            stream_pictures_capture(
                session=session, camera_id=camera_id, person_name=nome_pessoa
            ),
            media_type="multipart/x-mixed-replace;boundary=frame",
        )
    except Exception as e:
        raise e


# API endpoint to catch the atual image to a picture
@app.post("/capturar")
def capturar(session: Session = Depends(get_db)):
    try:
        set_capture_flag(session, 1)
        return 200, "Requisicao recebida"
    except Exception as e:
        raise e


# API endpoint to start background cameras
@app.get("/background_cameras")
def iniciar_cameras_background(session: Session = Depends(get_db)):
    camera = get_camera_by_id(session, 1)
    if camera is None:
        return {"status": "error", "message": "Camera não encontrada"}
    camera_ip = VideoCapture(
        f"rtsp://{camera.user}:{camera.password}@{camera.camera_ip}/"
    )
    while True:
        image_ok, frame = camera_ip.read()
        if image_ok:
            print("running on background")
        sleep(30)


# API endpoint for webcam-only photo capture (uses camera_id=0 as convention for webcam)
@app.get("/fotos/webcam/{nome_pessoa}")
def capturar_fotos_webcam(nome_pessoa: str, session: Session = Depends(get_db)):
    """
    Capture photos using local webcam (MacBook camera, etc).
    Uses camera_id=0 as convention for webcam mode.
    """
    try:
        return StreamingResponse(
            stream_pictures_capture(
                session=session, camera_id=0, person_name=nome_pessoa
            ),
            media_type="multipart/x-mixed-replace;boundary=frame",
        )
    except Exception as e:
        raise e


# API endpoint for webcam-only facial recognition
@app.get("/video/webcam")
def reconhecimento_facial_webcam(session: Session = Depends(get_db)):
    """
    Perform facial recognition using local webcam.
    Uses camera_id=0 as convention for webcam mode.
    """
    return StreamingResponse(
        stream_facial_recognition(session=session, id_camera=0),
        media_type="multipart/x-mixed-replace;boundary=frame",
    )


# API endpoint to list captured pictures
@app.get("/fotos")
def listar_fotos():
    """List all captured pictures."""
    pictures = list(PICTURES_DIR.glob("person.*.*.jpg"))
    return {"count": len(pictures), "pictures": [p.name for p in pictures]}


# API endpoint to delete all pictures (for retraining)
@app.delete("/fotos")
def deletar_fotos():
    """Delete all captured pictures."""
    pictures = list(PICTURES_DIR.glob("person.*.*.jpg"))
    deleted = 0
    for p in pictures:
        try:
            p.unlink()
            deleted += 1
        except Exception as e:
            print(f"Erro ao deletar {p}: {e}")
    return {"status": "success", "deleted": deleted}


# =============================================================================
# AUTO-CAPTURE ENDPOINTS (Recommended - no need to call /capturar)
# =============================================================================


@app.get("/captura-auto/webcam/{nome_pessoa}")
def captura_automatica_webcam(
    nome_pessoa: str,
    quantidade: int = Query(
        default=20, ge=1, le=100, description="Número de fotos a capturar"
    ),
    intervalo: float = Query(
        default=0.5, ge=0.1, le=5.0, description="Intervalo entre capturas em segundos"
    ),
    session: Session = Depends(get_db),
):
    """
    RECOMENDADO: Captura automática de fotos usando webcam.

    Abre o stream de vídeo e captura automaticamente quando detecta um rosto.
    Não precisa chamar /capturar - as fotos são salvas automaticamente.

    - **nome_pessoa**: Nome da pessoa sendo cadastrada
    - **quantidade**: Número de fotos a capturar (padrão: 20)
    - **intervalo**: Segundos entre cada captura (padrão: 0.5)

    Abra este endpoint no navegador para ver o vídeo em tempo real.
    """
    return StreamingResponse(
        stream_pictures_capture_auto(
            session=session,
            camera_id=0,
            person_name=nome_pessoa,
            samples_number=quantidade,
            capture_interval=intervalo,
        ),
        media_type="multipart/x-mixed-replace;boundary=frame",
    )


@app.get("/captura-auto/{camera_id}/{nome_pessoa}")
def captura_automatica_camera(
    camera_id: int,
    nome_pessoa: str,
    quantidade: int = Query(default=20, ge=1, le=100),
    intervalo: float = Query(default=0.5, ge=0.1, le=5.0),
    session: Session = Depends(get_db),
):
    """
    Captura automática de fotos usando câmera IP.

    Mesma funcionalidade do endpoint webcam, mas para câmeras IP cadastradas.
    """
    return StreamingResponse(
        stream_pictures_capture_auto(
            session=session,
            camera_id=camera_id,
            person_name=nome_pessoa,
            samples_number=quantidade,
            capture_interval=intervalo,
        ),
        media_type="multipart/x-mixed-replace;boundary=frame",
    )


# =============================================================================
# VIDEO FILE ANALYSIS ENDPOINTS
# =============================================================================


@app.post("/video/upload")
async def upload_video(file: UploadFile = File(...)):
    """
    Upload de arquivo de vídeo para análise posterior.

    Retorna o caminho do arquivo salvo para usar no endpoint de análise.
    """
    if not file.filename:
        return {"status": "error", "message": "Nenhum arquivo enviado"}

    # Validate file extension
    allowed_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        return {
            "status": "error",
            "message": f"Extensão não permitida. Use: {', '.join(allowed_extensions)}",
        }

    # Save file
    file_path = VIDEOS_DIR / file.filename
    try:
        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)

        return {
            "status": "success",
            "message": "Arquivo enviado com sucesso",
            "file_path": str(file_path),
            "file_name": file.filename,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/video/analisar-arquivo")
def analisar_video_stream(
    caminho: str = Query(..., description="Caminho do arquivo de vídeo"),
    session: Session = Depends(get_db),
):
    """
    Analisa um arquivo de vídeo e realiza reconhecimento facial.

    Retorna um stream de vídeo com as faces detectadas e identificadas.
    Abra no navegador para visualização em tempo real.

    - **caminho**: Caminho completo do arquivo de vídeo
    """
    return StreamingResponse(
        analyze_video_file(session=session, video_path=caminho),
        media_type="multipart/x-mixed-replace;boundary=frame",
    )


@app.post("/video/analisar-arquivo-json")
def analisar_video_json(
    caminho: str = Query(..., description="Caminho do arquivo de vídeo"),
    session: Session = Depends(get_db),
):
    """
    Analisa um arquivo de vídeo e retorna resultado em JSON.

    Útil para processamento programático sem visualização.

    - **caminho**: Caminho completo do arquivo de vídeo

    Retorna lista de pessoas reconhecidas com contagem de detecções.
    """
    return analyze_video_file_sync(session=session, video_path=caminho)


@app.get("/videos")
def listar_videos():
    """Lista todos os vídeos disponíveis para análise."""
    videos = []
    for ext in ["*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm"]:
        videos.extend(VIDEOS_DIR.glob(ext))

    return {
        "count": len(videos),
        "videos": [
            {
                "name": v.name,
                "path": str(v),
                "size_mb": round(v.stat().st_size / (1024 * 1024), 2),
            }
            for v in videos
        ],
    }


@app.delete("/videos/{filename}")
def deletar_video(filename: str):
    """Deleta um arquivo de vídeo."""
    file_path = VIDEOS_DIR / filename
    if not file_path.exists():
        return {"status": "error", "message": "Arquivo não encontrado"}

    try:
        file_path.unlink()
        return {"status": "success", "message": f"Arquivo {filename} deletado"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# =============================================================================
# HTML INTERFACE FOR CAPTURE WITH REAL-TIME VIDEO
# =============================================================================


@app.get("/captura", response_class=HTMLResponse)
def pagina_captura(session: Session = Depends(get_db)):
    """
    Página HTML para captura de fotos com visualização em tempo real.
    Acesse no navegador para usar a interface gráfica.
    """
    html_content = """
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Captura de Fotos - Reconhecimento Facial</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                min-height: 100vh;
                color: white;
                padding: 20px;
            }
            .container { max-width: 1200px; margin: 0 auto; }
            h1 { text-align: center; margin-bottom: 20px; color: #00d4ff; }
            .info-bar {
                background: rgba(0, 212, 255, 0.1);
                border: 1px solid #00d4ff;
                border-radius: 10px;
                padding: 15px;
                margin-bottom: 20px;
                display: flex;
                justify-content: space-between;
                flex-wrap: wrap;
                gap: 10px;
            }
            .main-content {
                display: grid;
                grid-template-columns: 1fr 350px;
                gap: 20px;
            }
            @media (max-width: 900px) {
                .main-content { grid-template-columns: 1fr; }
            }
            .video-container {
                background: #000;
                border-radius: 10px;
                overflow: hidden;
            }
            .video-container img {
                width: 100%;
                height: auto;
                display: block;
            }
            .controls {
                background: rgba(255, 255, 255, 0.05);
                border-radius: 10px;
                padding: 20px;
            }
            .section { margin-bottom: 20px; }
            .section h3 {
                color: #00d4ff;
                margin-bottom: 10px;
                font-size: 14px;
                text-transform: uppercase;
            }
            .btn {
                width: 100%;
                padding: 15px;
                border: none;
                border-radius: 8px;
                font-size: 16px;
                font-weight: bold;
                cursor: pointer;
                margin-bottom: 10px;
                transition: all 0.2s;
            }
            .btn:hover { transform: scale(1.02); }
            .btn:active { transform: scale(0.98); }
            .btn-primary {
                background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%);
                color: white;
            }
            .btn-success {
                background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%);
                color: white;
            }
            .btn-secondary {
                background: rgba(255, 255, 255, 0.1);
                color: white;
                border: 1px solid rgba(255, 255, 255, 0.2);
            }
            .btn-danger {
                background: linear-gradient(135deg, #f44336 0%, #c62828 100%);
                color: white;
            }
            input[type="text"] {
                width: 100%;
                padding: 12px;
                border: 1px solid #444;
                border-radius: 8px;
                background: rgba(255,255,255,0.1);
                color: white;
                font-size: 16px;
                margin-bottom: 10px;
            }
            .counter {
                text-align: center;
                padding: 20px;
                background: rgba(0, 212, 255, 0.1);
                border-radius: 10px;
                margin-bottom: 20px;
            }
            .counter .number {
                font-size: 48px;
                font-weight: bold;
                color: #00d4ff;
            }
            .status {
                padding: 10px;
                border-radius: 5px;
                text-align: center;
                margin-top: 10px;
            }
            .status.success { background: rgba(76, 175, 80, 0.2); color: #4CAF50; }
            .status.error { background: rgba(244, 67, 54, 0.2); color: #f44336; }
            .status.info { background: rgba(0, 212, 255, 0.2); color: #00d4ff; }
            .instructions {
                background: rgba(255, 193, 7, 0.1);
                border: 1px solid #ffc107;
                border-radius: 8px;
                padding: 15px;
                margin-bottom: 20px;
            }
            .instructions h4 { color: #ffc107; margin-bottom: 10px; }
            .instructions li { margin-bottom: 5px; color: #ddd; margin-left: 20px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📷 Captura de Fotos - Reconhecimento Facial</h1>
            
            <div class="instructions">
                <h4>📋 Instruções</h4>
                <ol>
                    <li>Digite o nome da pessoa e clique em "Iniciar Sessão"</li>
                    <li>Posicione o rosto na câmera (quadrado verde = rosto detectado)</li>
                    <li>Clique em "CAPTURAR" ou pressione ESPAÇO</li>
                    <li>Capture pelo menos 20 fotos em diferentes ângulos</li>
                    <li>Clique em "Treinar Modelo" quando terminar</li>
                </ol>
            </div>
            
            <div class="main-content">
                <div class="video-container">
                    <div id="modeIndicator" style="background: #ff6b6b; color: white; padding: 5px 15px; text-align: center; font-weight: bold;">
                        MODO: RECONHECIMENTO
                    </div>
                    <img id="videoStream" src="/stream/reconhecimento" alt="Video Stream">
                </div>
                
                <div class="controls">
                    <div class="section">
                        <h3>Modo Atual</h3>
                        <button class="btn btn-primary" id="btnModeRecog" onclick="setModeRecognition()" style="background: #ff6b6b;">
                            👁️ Reconhecimento
                        </button>
                        <button class="btn btn-secondary" id="btnModeCapture" onclick="setModeCapture()">
                            📷 Captura
                        </button>
                    </div>
                    
                    <div class="section">
                        <h3>Iniciar Sessão de Captura</h3>
                        <input type="text" id="personName" placeholder="Nome da pessoa">
                        <button class="btn btn-success" onclick="startSession()">
                            ▶️ Iniciar Sessão
                        </button>
                    </div>
                    
                    <div class="counter">
                        <div class="number" id="photoCount">0</div>
                        <div>fotos capturadas</div>
                        <div id="sessionInfo" style="font-size: 12px; color: #aaa; margin-top: 5px;"></div>
                    </div>
                    
                    <div class="section" id="captureSection" style="display: none;">
                        <h3>Capturar Fotos</h3>
                        <button class="btn btn-primary" id="btnCapture" onclick="capturePhoto()">
                            📸 CAPTURAR (Espaço)
                        </button>
                        <button class="btn btn-secondary" onclick="captureMultiple(5)" id="btnMulti">
                            ⚡ Capturar 5 automático
                        </button>
                    </div>
                    
                    <div class="section">
                        <h3>Ações</h3>
                        <button class="btn btn-success" onclick="trainModel()">
                            🧠 Treinar Modelo
                        </button>
                        <button class="btn btn-secondary" onclick="checkStatus()">📊 Status</button>
                        <button class="btn btn-danger" onclick="resetSession()">🔄 Resetar</button>
                    </div>
                    
                    <div id="statusMessage" class="status info" style="display: none;"></div>
                </div>
            </div>
        </div>

        <script>
            let isSessionActive = false;
            let currentMode = 'recognition';
            
            function showStatus(msg, type) {
                const el = document.getElementById('statusMessage');
                el.textContent = msg;
                el.className = 'status ' + type;
                el.style.display = 'block';
                setTimeout(() => el.style.display = 'none', 4000);
            }
            
            function setModeRecognition() {
                currentMode = 'recognition';
                document.getElementById('videoStream').src = '/stream/reconhecimento';
                document.getElementById('modeIndicator').textContent = 'MODO: RECONHECIMENTO';
                document.getElementById('modeIndicator').style.background = '#ff6b6b';
                document.getElementById('btnModeRecog').style.background = '#ff6b6b';
                document.getElementById('btnModeRecog').classList.remove('btn-secondary');
                document.getElementById('btnModeCapture').style.background = '';
                document.getElementById('btnModeCapture').classList.add('btn-secondary');
                document.getElementById('captureSection').style.display = 'none';
                showStatus('Modo reconhecimento ativado', 'info');
            }
            
            function setModeCapture() {
                currentMode = 'capture';
                document.getElementById('videoStream').src = '/stream/video/0';
                document.getElementById('modeIndicator').textContent = 'MODO: CAPTURA';
                document.getElementById('modeIndicator').style.background = '#00d4ff';
                document.getElementById('btnModeCapture').style.background = '#00d4ff';
                document.getElementById('btnModeCapture').classList.remove('btn-secondary');
                document.getElementById('btnModeRecog').style.background = '';
                document.getElementById('btnModeRecog').classList.add('btn-secondary');
                if (isSessionActive) {
                    document.getElementById('captureSection').style.display = 'block';
                }
                showStatus('Modo captura ativado', 'info');
            }
            
            async function startSession() {
                const name = document.getElementById('personName').value.trim();
                if (!name) {
                    showStatus('Digite o nome da pessoa', 'error');
                    return;
                }
                try {
                    const resp = await fetch('/captura/iniciar/' + encodeURIComponent(name), {method: 'POST'});
                    const data = await resp.json();
                    if (data.status === 'success') {
                        showStatus('Sessão iniciada para: ' + name, 'success');
                        document.getElementById('sessionInfo').textContent = 'Pessoa: ' + name + ' (ID: ' + data.person_id + ')';
                        isSessionActive = true;
                        // Switch to capture mode automatically
                        setModeCapture();
                        document.getElementById('captureSection').style.display = 'block';
                        updateCount();
                    } else {
                        showStatus(data.message, 'error');
                    }
                } catch(e) {
                    showStatus('Erro: ' + e.message, 'error');
                }
            }
            
            async function capturePhoto() {
                if (!isSessionActive) {
                    showStatus('Inicie uma sessão primeiro', 'error');
                    return;
                }
                if (currentMode !== 'capture') {
                    showStatus('Mude para modo captura primeiro', 'error');
                    return;
                }
                try {
                    const resp = await fetch('/captura/foto', {method: 'POST'});
                    const data = await resp.json();
                    showStatus(data.message, data.status === 'success' ? 'success' : 'error');
                    updateCount();
                } catch(e) {
                    showStatus('Erro: ' + e.message, 'error');
                }
            }
            
            async function captureMultiple(n) {
                for (let i = 0; i < n; i++) {
                    await capturePhoto();
                    await new Promise(r => setTimeout(r, 600));
                }
            }
            
            async function updateCount() {
                try {
                    const resp = await fetch('/captura/estado');
                    const data = await resp.json();
                    document.getElementById('photoCount').textContent = data.samples_captured || 0;
                } catch(e) {}
            }
            
            async function trainModel() {
                showStatus('Treinando modelo...', 'info');
                try {
                    const resp = await fetch('/treinamento');
                    const data = await resp.json();
                    showStatus(data.message, data.status === 'success' ? 'success' : 'error');
                } catch(e) {
                    showStatus('Erro: ' + e.message, 'error');
                }
            }
            
            async function resetSession() {
                try {
                    await fetch('/captura/resetar', {method: 'POST'});
                    document.getElementById('photoCount').textContent = '0';
                    document.getElementById('sessionInfo').textContent = '';
                    document.getElementById('btnCapture').disabled = true;
                    document.getElementById('btnMulti').disabled = true;
                    isSessionActive = false;
                    showStatus('Sessão resetada', 'info');
                } catch(e) {}
            }
            
            async function checkStatus() {
                try {
                    const resp = await fetch('/status');
                    const data = await resp.json();
                    showStatus('Fotos: ' + data.pictures_count + ' | Modelo: ' + (data.classifier_trained ? 'Treinado' : 'Não'), 'info');
                } catch(e) {}
            }
            
            // Keyboard shortcut
            document.addEventListener('keydown', (e) => {
                if (e.code === 'Space' && isSessionActive) {
                    e.preventDefault();
                    capturePhoto();
                }
            });
            
            // Update count periodically
            setInterval(updateCount, 2000);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/stream/video/{camera_id}")
def stream_video(camera_id: int):
    """Stream de vídeo com detecção facial (sem captura automática)."""
    return StreamingResponse(
        stream_video_only(camera_id=camera_id),
        media_type="multipart/x-mixed-replace;boundary=frame",
    )


@app.get("/stream/reconhecimento")
def stream_reconhecimento():
    """Stream de reconhecimento facial sem dependência de banco de dados."""
    return StreamingResponse(
        stream_recognition_only(),
        media_type="multipart/x-mixed-replace;boundary=frame",
    )


@app.post("/captura/iniciar/{nome_pessoa}")
def iniciar_sessao_captura(nome_pessoa: str, session: Session = Depends(get_db)):
    """Inicia uma sessão de captura para uma pessoa e registra no banco."""
    from schema import CreateAndUpdatePerson

    # Get next available person_id from database
    persons = get_all_persons(session)
    existing_ids = {p.person_id for p in persons}
    person_id = max(existing_ids, default=0) + 1

    # Register person in database immediately
    try:
        person = CreateAndUpdatePerson(person_id=person_id, name=nome_pessoa)
        create_person(session=session, person_info=person)
    except Exception as e:
        print(f"Error registering person: {e}")

    start_capture_session(person_id, nome_pessoa, 20)
    return {
        "status": "success",
        "message": f"Sessão iniciada para {nome_pessoa}",
        "person_id": person_id,
        "person_name": nome_pessoa,
    }


@app.post("/captura/foto")
def capturar_foto():
    """Captura uma foto (deve estar com sessão ativa e stream aberto)."""
    state = get_capture_state()
    if not state["is_active"]:
        return {
            "status": "error",
            "message": "Nenhuma sessão ativa. Inicie com /captura/iniciar/{nome}",
        }

    if state["samples_captured"] >= state["max_samples"]:
        return {"status": "error", "message": "Limite de fotos atingido"}

    trigger_capture()
    return {
        "status": "success",
        "message": f"Captura solicitada ({state['samples_captured'] + 1}/{state['max_samples']})",
    }


@app.get("/captura/estado")
def estado_captura():
    """Retorna o estado atual da captura."""
    return get_capture_state()


@app.post("/captura/resetar")
def resetar_captura():
    """Reseta o estado da captura."""
    reset_capture_state()
    return {"status": "success", "message": "Estado resetado"}


@app.post("/captura/finalizar")
def finalizar_captura():
    """Finaliza a sessão de captura (pessoa já foi registrada ao iniciar)."""
    state = get_capture_state()
    if not state["is_active"]:
        return {"status": "error", "message": "Nenhuma sessão ativa"}

    samples = state["samples_captured"]
    person_name = state["person_name"]
    reset_capture_state()

    return {
        "status": "success",
        "message": f"Sessão finalizada para {person_name} com {samples} fotos",
        "samples_captured": samples,
    }


# =============================================================================
# HTML INTERFACE FOR VIDEO FILE ANALYSIS
# =============================================================================


@app.get("/analise", response_class=HTMLResponse)
def pagina_analise_video():
    """
    Página HTML para análise de arquivos de vídeo com reconhecimento facial.
    Acesse no navegador para usar a interface gráfica.
    """
    html_content = """
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Análise de Vídeos - Reconhecimento Facial</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                min-height: 100vh;
                color: white;
                padding: 20px;
            }
            .container { max-width: 1200px; margin: 0 auto; }
            h1 { text-align: center; margin-bottom: 20px; color: #ff6b6b; }
            .main-content {
                display: grid;
                grid-template-columns: 1fr 350px;
                gap: 20px;
            }
            @media (max-width: 900px) {
                .main-content { grid-template-columns: 1fr; }
            }
            .video-container {
                background: #000;
                border-radius: 10px;
                overflow: hidden;
                min-height: 400px;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            .video-container img {
                width: 100%;
                height: auto;
                display: block;
            }
            .video-container .placeholder {
                color: #666;
                text-align: center;
            }
            .controls {
                background: rgba(255, 255, 255, 0.05);
                border-radius: 10px;
                padding: 20px;
            }
            .section { margin-bottom: 20px; }
            .section h3 {
                color: #ff6b6b;
                margin-bottom: 10px;
                font-size: 14px;
                text-transform: uppercase;
            }
            .btn {
                width: 100%;
                padding: 15px;
                border: none;
                border-radius: 8px;
                font-size: 16px;
                font-weight: bold;
                cursor: pointer;
                margin-bottom: 10px;
                transition: all 0.2s;
            }
            .btn:hover { transform: scale(1.02); }
            .btn:active { transform: scale(0.98); }
            .btn:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
            .btn-primary {
                background: linear-gradient(135deg, #ff6b6b 0%, #c0392b 100%);
                color: white;
            }
            .btn-success {
                background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%);
                color: white;
            }
            .btn-secondary {
                background: rgba(255, 255, 255, 0.1);
                color: white;
                border: 1px solid rgba(255, 255, 255, 0.2);
            }
            .file-input-wrapper {
                position: relative;
                overflow: hidden;
                display: block;
            }
            .file-input-wrapper input[type=file] {
                position: absolute;
                left: 0;
                top: 0;
                opacity: 0;
                width: 100%;
                height: 100%;
                cursor: pointer;
            }
            .video-list {
                max-height: 200px;
                overflow-y: auto;
                background: rgba(0,0,0,0.3);
                border-radius: 8px;
                padding: 10px;
            }
            .video-item {
                padding: 10px;
                background: rgba(255,255,255,0.05);
                border-radius: 5px;
                margin-bottom: 5px;
                cursor: pointer;
                transition: background 0.2s;
            }
            .video-item:hover { background: rgba(255,255,255,0.1); }
            .video-item.selected { background: rgba(255, 107, 107, 0.3); border: 1px solid #ff6b6b; }
            .video-item .name { font-weight: bold; }
            .video-item .size { font-size: 12px; color: #aaa; }
            .results {
                background: rgba(0,0,0,0.3);
                border-radius: 8px;
                padding: 15px;
                max-height: 300px;
                overflow-y: auto;
            }
            .results h4 { color: #4CAF50; margin-bottom: 10px; }
            .person-result {
                padding: 8px;
                background: rgba(255,255,255,0.05);
                border-radius: 5px;
                margin-bottom: 5px;
            }
            .person-result .name { color: #00d4ff; font-weight: bold; }
            .status {
                padding: 10px;
                border-radius: 5px;
                text-align: center;
                margin-top: 10px;
            }
            .status.success { background: rgba(76, 175, 80, 0.2); color: #4CAF50; }
            .status.error { background: rgba(244, 67, 54, 0.2); color: #f44336; }
            .status.info { background: rgba(0, 212, 255, 0.2); color: #00d4ff; }
            .instructions {
                background: rgba(255, 107, 107, 0.1);
                border: 1px solid #ff6b6b;
                border-radius: 8px;
                padding: 15px;
                margin-bottom: 20px;
            }
            .instructions h4 { color: #ff6b6b; margin-bottom: 10px; }
            .instructions li { margin-bottom: 5px; color: #ddd; margin-left: 20px; }
            .nav-links {
                display: flex;
                gap: 10px;
                margin-bottom: 20px;
            }
            .nav-links a {
                color: #00d4ff;
                text-decoration: none;
                padding: 8px 15px;
                background: rgba(0, 212, 255, 0.1);
                border-radius: 5px;
            }
            .nav-links a:hover { background: rgba(0, 212, 255, 0.2); }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="nav-links">
                <a href="/captura">📷 Captura de Fotos</a>
                <a href="/analise">🎬 Análise de Vídeos</a>
                <a href="/docs">📚 API Docs</a>
            </div>
            
            <h1>🎬 Análise de Vídeos - Reconhecimento Facial</h1>
            
            <div class="instructions">
                <h4>📋 Instruções</h4>
                <ol>
                    <li>Primeiro, capture fotos e treine o modelo em <a href="/captura" style="color:#00d4ff">/captura</a></li>
                    <li>Faça upload de um arquivo de vídeo (.mp4, .avi, .mov)</li>
                    <li>Selecione o vídeo na lista</li>
                    <li>Clique em "Analisar com Vídeo" para ver em tempo real</li>
                    <li>Ou clique em "Analisar (JSON)" para resultado rápido</li>
                </ol>
            </div>
            
            <div class="main-content">
                <div class="video-container" id="videoContainer">
                    <div class="placeholder">
                        <p>📹 Selecione um vídeo e clique em "Analisar"</p>
                        <p style="font-size: 12px; margin-top: 10px;">O vídeo será processado com reconhecimento facial</p>
                    </div>
                </div>
                
                <div class="controls">
                    <div class="section">
                        <h3>1. Upload de Vídeo</h3>
                        <div class="file-input-wrapper">
                            <button class="btn btn-secondary">📁 Selecionar Arquivo</button>
                            <input type="file" id="videoFile" accept=".mp4,.avi,.mov,.mkv,.webm" onchange="uploadVideo(this)">
                        </div>
                        <div id="uploadStatus" style="font-size: 12px; color: #aaa; margin-top: 5px;"></div>
                    </div>
                    
                    <div class="section">
                        <h3>2. Vídeos Disponíveis</h3>
                        <div class="video-list" id="videoList">
                            <div style="color: #666; text-align: center;">Carregando...</div>
                        </div>
                        <button class="btn btn-secondary" onclick="loadVideos()" style="margin-top: 10px;">
                            🔄 Atualizar Lista
                        </button>
                    </div>
                    
                    <div class="section">
                        <h3>3. Analisar</h3>
                        <button class="btn btn-primary" id="btnAnalyze" onclick="analyzeVideoStream()" disabled>
                            🎬 Analisar com Vídeo
                        </button>
                        <button class="btn btn-success" id="btnAnalyzeJson" onclick="analyzeVideoJson()" disabled>
                            📊 Analisar (JSON Rápido)
                        </button>
                    </div>
                    
                    <div class="section">
                        <h3>Resultados</h3>
                        <div class="results" id="results">
                            <p style="color: #666; text-align: center;">Nenhuma análise realizada</p>
                        </div>
                    </div>
                    
                    <div class="section">
                        <h3>Ações</h3>
                        <button class="btn btn-secondary" onclick="checkStatus()">📊 Status do Sistema</button>
                        <button class="btn btn-secondary" onclick="deleteSelectedVideo()">🗑️ Deletar Vídeo</button>
                    </div>
                    
                    <div id="statusMessage" class="status info" style="display: none;"></div>
                </div>
            </div>
        </div>

        <script>
            let selectedVideo = null;
            
            function showStatus(msg, type) {
                const el = document.getElementById('statusMessage');
                el.textContent = msg;
                el.className = 'status ' + type;
                el.style.display = 'block';
                setTimeout(() => el.style.display = 'none', 4000);
            }
            
            async function uploadVideo(input) {
                const file = input.files[0];
                if (!file) return;
                
                document.getElementById('uploadStatus').textContent = 'Enviando...';
                
                const formData = new FormData();
                formData.append('file', file);
                
                try {
                    const resp = await fetch('/video/upload', {
                        method: 'POST',
                        body: formData
                    });
                    const data = await resp.json();
                    if (data.status === 'success') {
                        document.getElementById('uploadStatus').textContent = 'Upload concluído!';
                        showStatus('Vídeo enviado: ' + file.name, 'success');
                        loadVideos();
                    } else {
                        document.getElementById('uploadStatus').textContent = 'Erro: ' + data.message;
                        showStatus(data.message, 'error');
                    }
                } catch(e) {
                    document.getElementById('uploadStatus').textContent = 'Erro no upload';
                    showStatus('Erro: ' + e.message, 'error');
                }
            }
            
            async function loadVideos() {
                try {
                    const resp = await fetch('/videos');
                    const data = await resp.json();
                    const list = document.getElementById('videoList');
                    
                    if (data.videos.length === 0) {
                        list.innerHTML = '<div style="color: #666; text-align: center;">Nenhum vídeo disponível</div>';
                        return;
                    }
                    
                    list.innerHTML = data.videos.map(v => `
                        <div class="video-item" onclick="selectVideo('${v.path}', '${v.name}')" data-path="${v.path}">
                            <div class="name">📹 ${v.name}</div>
                            <div class="size">${v.size_mb} MB</div>
                        </div>
                    `).join('');
                } catch(e) {
                    showStatus('Erro ao carregar vídeos', 'error');
                }
            }
            
            function selectVideo(path, name) {
                selectedVideo = { path, name };
                document.querySelectorAll('.video-item').forEach(el => el.classList.remove('selected'));
                document.querySelector(`[data-path="${path}"]`).classList.add('selected');
                document.getElementById('btnAnalyze').disabled = false;
                document.getElementById('btnAnalyzeJson').disabled = false;
                showStatus('Selecionado: ' + name, 'info');
            }
            
            function analyzeVideoStream() {
                if (!selectedVideo) {
                    showStatus('Selecione um vídeo primeiro', 'error');
                    return;
                }
                
                const container = document.getElementById('videoContainer');
                container.innerHTML = `<img src="/video/analisar-arquivo?caminho=${encodeURIComponent(selectedVideo.path)}" alt="Análise de Vídeo">`;
                showStatus('Iniciando análise...', 'info');
            }
            
            async function analyzeVideoJson() {
                if (!selectedVideo) {
                    showStatus('Selecione um vídeo primeiro', 'error');
                    return;
                }
                
                showStatus('Analisando vídeo...', 'info');
                document.getElementById('results').innerHTML = '<p style="color: #00d4ff; text-align: center;">⏳ Processando...</p>';
                
                try {
                    const resp = await fetch(`/video/analisar-arquivo-json?caminho=${encodeURIComponent(selectedVideo.path)}`, {
                        method: 'POST'
                    });
                    const data = await resp.json();
                    
                    if (data.status === 'error') {
                        document.getElementById('results').innerHTML = `<p style="color: #f44336;">${data.message}</p>`;
                        showStatus(data.message, 'error');
                        return;
                    }
                    
                    let html = `
                        <h4>✅ Análise Concluída</h4>
                        <p><strong>Arquivo:</strong> ${selectedVideo.name}</p>
                        <p><strong>Frames:</strong> ${data.total_frames} (${data.frames_processed} processados)</p>
                        <p><strong>Faces detectadas:</strong> ${data.faces_detected}</p>
                        <hr style="border-color: #333; margin: 10px 0;">
                        <p><strong>Pessoas reconhecidas:</strong></p>
                    `;
                    
                    if (data.recognized_persons && data.recognized_persons.length > 0) {
                        data.recognized_persons.forEach(p => {
                            html += `
                                <div class="person-result">
                                    <span class="name">${p.name}</span>
                                    <span style="float: right;">${p.detections} detecções</span>
                                </div>
                            `;
                        });
                    } else {
                        html += '<p style="color: #aaa;">Nenhuma pessoa reconhecida</p>';
                    }
                    
                    document.getElementById('results').innerHTML = html;
                    showStatus('Análise concluída!', 'success');
                } catch(e) {
                    showStatus('Erro: ' + e.message, 'error');
                }
            }
            
            async function deleteSelectedVideo() {
                if (!selectedVideo) {
                    showStatus('Selecione um vídeo primeiro', 'error');
                    return;
                }
                
                if (!confirm('Deletar ' + selectedVideo.name + '?')) return;
                
                try {
                    const resp = await fetch(`/videos/${selectedVideo.name}`, { method: 'DELETE' });
                    const data = await resp.json();
                    showStatus(data.message, data.status === 'success' ? 'success' : 'error');
                    loadVideos();
                    selectedVideo = null;
                } catch(e) {
                    showStatus('Erro: ' + e.message, 'error');
                }
            }
            
            async function checkStatus() {
                try {
                    const resp = await fetch('/status');
                    const data = await resp.json();
                    showStatus(`Fotos: ${data.pictures_count} | Modelo: ${data.classifier_trained ? 'Treinado ✅' : 'Não treinado ❌'}`, 'info');
                } catch(e) {
                    showStatus('Erro ao verificar status', 'error');
                }
            }
            
            // Load videos on page load
            loadVideos();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)
