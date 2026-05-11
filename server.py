import os
import json
import time
import uuid
import random
import asyncio
import threading
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify, Response, send_from_directory, send_file
from flask_cors import CORS
from openai import OpenAI
import torch
import numpy as np
import cv2
from PIL import Image
import traceback
import sys
from omegaconf import OmegaConf

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "models" / "facebook" / "sam-3d-body-dinov3"))
sys.path.insert(0, str(BASE_DIR / "models" / "facebook" / "sam-3d-objects"))
sys.path.insert(0, str(BASE_DIR / "dinov3"))
sys.path.insert(0, str(BASE_DIR / "dinov3-vith16plus-pretrain-lvd1689m"))


os.environ['TORCH_HOME'] = str(BASE_DIR / "models" / "dinov3")
os.environ['TORCH_HUB_DIR'] = str(BASE_DIR / "models" / "dinov3")

# 设置环境变量跳过初始化
os.environ['LIDRA_SKIP_INIT'] = '1'

from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
from sam_3d_body.utils.mhr_export import export_mhr, export_mhr_batch, export_mhr_combined

app = Flask(__name__)
CORS(app)

OUTPUT_DIR = BASE_DIR / "output"

# 尝试多个可能的ComfyUI输出目录
COMFYUI_OUTPUT_DIR = BASE_DIR / "ComfyUI_windows_portable" / "ComfyUI" / "output"

# 如果上面的目录不存在，尝试其他可能的路径
if not COMFYUI_OUTPUT_DIR.exists():
    # 尝试父目录的ComfyUI输出
    alt_dir = BASE_DIR.parent / "ComfyUI_windows_portable" / "ComfyUI" / "output"
    if alt_dir.exists():
        COMFYUI_OUTPUT_DIR = alt_dir
        print(f"使用ComfyUI输出目录: {alt_dir}")
    else:
        # 尝试当前目录的output
        alt_dir = BASE_DIR / "output"
        if alt_dir.exists():
            COMFYUI_OUTPUT_DIR = alt_dir
            print(f"使用ComfyUI输出目录: {alt_dir}")

print(f"BASE_DIR: {BASE_DIR}")
print(f"COMFYUI_OUTPUT_DIR: {COMFYUI_OUTPUT_DIR}")
print(f"目录是否存在: {COMFYUI_OUTPUT_DIR.exists()}")

WORKFLOWS = {
    "nunchaku_z_image_turbo": BASE_DIR / "nunchaku-z-image-turbo.json",
    "flux2_klein": BASE_DIR / "flux2-klein文生图工作流.json",
    "flux2_klein_9b": BASE_DIR / "Flux2_Klein+文生图9B版-官方版-4步.json",
    "flux2_klein_edit": BASE_DIR / "Flux2++Klein图像单图编辑9B版官方版.json"
}

RESOLUTIONS = {
    "1920x1080": (1920, 1080),
    "1080x1920": (1080, 1920),
    "1024x1024": (1024, 1024),
    "1280x720": (1280, 720),
    "720x1280": (720, 1280)
}

PROJECTS = {}

DEFAULT_API_KEYS = {
    "deepseek": "sk-6c948d3432164f60bb09e3b1f1708af6"
}

class LLMManager:
    def __init__(self):
        self.clients = {}
    
    def get_client(self, provider, base_url, api_key=None):
        # 标准化 base_url
        base_url = base_url.rstrip('/')
        
        # 对于需要 OpenAI 兼容 API 的提供商，自动补充 /v1
        if provider != "deepseek" and not base_url.endswith('/v1'):
            base_url = base_url + '/v1'
        
        if provider == "ollama":
            return OpenAI(base_url=base_url, api_key="ollama")
        elif provider == "lmstudio":
            return OpenAI(base_url=base_url, api_key="lmstudio")
        elif provider == "vllm":
            return OpenAI(base_url=base_url, api_key="vllm")
        elif provider == "deepseek":
            # DeepSeek 不需要 /v1
            api_key = api_key or DEFAULT_API_KEYS.get("deepseek", "")
            return OpenAI(base_url="https://api.deepseek.com", api_key=api_key)
        elif provider == "siliconflow":
            return OpenAI(base_url=base_url, api_key=api_key)
        elif provider == "custom":
            return OpenAI(base_url=base_url, api_key=api_key or "custom")
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    def get_models(self, provider, base_url, api_key=None):
        try:
            client = self.get_client(provider, base_url, api_key)
            models = client.models.list()
            return [model.id for model in models.data]
        except Exception as e:
            print(f"Error getting models: {e}")
            traceback.print_exc()
            return []
    
    def stream_chat(self, provider, base_url, model, messages, api_key=None):
        try:
            client = self.get_client(provider, base_url, api_key)
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True
            )
            for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            print(f"LLM stream chat error: {e}")
            traceback.print_exc()
            raise Exception(f"LLM connection error: {str(e)}")

llm_manager = LLMManager()

class ComfyUIManager:
    def __init__(self):
        self.base_url = "http://127.0.0.1:8188"
        self.client_id = str(uuid.uuid4())
    
    def load_workflow(self, workflow_name):
        workflow_path = WORKFLOWS.get(workflow_name)
        if not workflow_path or not workflow_path.exists():
            raise ValueError(f"Workflow not found: {workflow_name}")
        with open(workflow_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def modify_workflow(self, workflow, prompt, negative_prompt="", width=1024, height=1024, seed=None, image_name=None):
        workflow_data = json.loads(json.dumps(workflow))

        print(f"=== 修改工作流 ===")
        print(f"Prompt: {prompt[:100]}...")
        print(f"Negative: {negative_prompt[:50] if negative_prompt else 'None'}")
        print(f"尺寸: {width}x{height}")
        print(f"Seed: {seed}")

        # 收集CLIPTextEncode节点
        clip_encode_nodes = []
        for node_id, node in workflow_data.items():
            if node.get("class_type") == "CLIPTextEncode":
                clip_encode_nodes.append((node_id, node))
                print(f"找到CLIPTextEncode节点: {node_id}, 当前文本: {node.get('inputs', {}).get('text', 'None')[:50]}...")
            elif node.get("class_type") == "EmptySD3LatentImage":
                node["inputs"]["width"] = width
                node["inputs"]["height"] = height
                print(f"设置EmptySD3LatentImage节点 {node_id} 尺寸为 {width}x{height}")
            elif node.get("class_type") == "EmptyFlux2LatentImage":
                node["inputs"]["width"] = width
                node["inputs"]["height"] = height
                print(f"设置EmptyFlux2LatentImage节点 {node_id} 尺寸为 {width}x{height}")
            elif node.get("class_type") == "KSampler" and seed is not None:
                node["inputs"]["seed"] = seed
                print(f"设置KSampler节点 {node_id} 的seed为 {seed}")
            elif node.get("class_type") == "LoadImage" and image_name is not None:
                node["inputs"]["image"] = image_name
                print(f"设置LoadImage节点 {node_id} 的图片为 {image_name}")

        # 设置CLIPTextEncode节点的文本
        # 如果有2个CLIPTextEncode节点，第一个是negative，第二个是positive
        # 如果只有1个，那就是positive（nunchaku工作流）
        if len(clip_encode_nodes) == 2:
            clip_encode_nodes[0][1]["inputs"]["text"] = negative_prompt
            clip_encode_nodes[1][1]["inputs"]["text"] = prompt
            print(f"✅ 设置negative提示词到节点 {clip_encode_nodes[0][0]}")
            print(f"✅ 设置positive提示词到节点 {clip_encode_nodes[1][0]}")
        elif len(clip_encode_nodes) == 1:
            clip_encode_nodes[0][1]["inputs"]["text"] = prompt
            print(f"✅ 设置positive提示词到节点 {clip_encode_nodes[0][0]} (无negative)")
        else:
            print(f"⚠️ 警告: 找到 {len(clip_encode_nodes)} 个CLIPTextEncode节点")

        return workflow_data
    
    def upload_image(self, image_path):
        import requests
        with open(image_path, 'rb') as f:
            files = {'image': f}
            response = requests.post(f"{self.base_url}/upload/image", files=files, timeout=60)
        return response.json()
    
    def queue_prompt(self, workflow, client_id=None):
        import requests
        # 如果没有传入client_id，使用实例的client_id（兼容旧代码）
        cid = client_id if client_id is not None else self.client_id
        prompt = {"prompt": workflow, "client_id": cid}
        print(f"Queue prompt to ComfyUI with client_id: {cid}")

        try:
            response = requests.post(f"{self.base_url}/prompt", json=prompt, timeout=30)
            result = response.json()
            print(f"Queue response: {result}")
            return result
        except requests.exceptions.Timeout:
            print("❌ ComfyUI请求超时")
            raise TimeoutError("ComfyUI请求超时")
        except Exception as e:
            print(f"❌ ComfyUI请求失败: {e}")
            raise

    def get_history(self, prompt_id, timeout=10):
        import requests
        try:
            response = requests.get(f"{self.base_url}/history/{prompt_id}", timeout=timeout)
            return response.json()
        except requests.exceptions.Timeout:
            print(f"⚠️ get_history 请求超时（{timeout}秒）")
            return {}
        except Exception as e:
            print(f"⚠️ get_history 请求失败: {e}")
            return {}

    def wait_for_completion(self, prompt_id, client_id=None, timeout=600):
        import requests
        print(f"⏳ 等待ComfyUI执行完成，prompt_id={prompt_id}，超时={timeout}秒")

        start_time = time.time()
        poll_interval = 0.3
        max_attempts = int(timeout / poll_interval)
        max_consecutive_errors = 20
        consecutive_errors = 0  # 初始化连续错误计数器
        initial_delay = 5  # 增加初始延迟，给ComfyUI更多时间开始处理（特别是Flux2工作流）

        print(f"⏳ 初始等待 {initial_delay} 秒，让ComfyUI开始处理...")
        time.sleep(initial_delay)
        
        for attempt in range(max_attempts):
            try:
                history = self.get_history(prompt_id)
                
                if not history:
                    consecutive_errors += 1
                    elapsed = time.time() - start_time
                    print(f"⚠️ get_history 返回空（连续错误{consecutive_errors}/{max_consecutive_errors}，已等待{elapsed:.1f}秒）")
                    # 不立即失败，继续尝试，给Flux2等工作流更多时间
                    if consecutive_errors >= max_consecutive_errors:
                        print(f"❌ ComfyUI 连续 {max_consecutive_errors} 次无响应")
                        print(f"   可能原因：")
                        print(f"   1. ComfyUI 服务未正常运行")
                        print(f"   2. ComfyUI 的 history API 出现问题（可能工作流执行时间过长）")
                        print(f"   3. 网络连接问题")
                        print(f"   4. Prompt ID {prompt_id} 可能无效")
                        print(f"   5. 工作流可能仍在后台执行，建议检查ComfyUI界面")
                        raise RuntimeError(f"ComfyUI 连续 {max_consecutive_errors} 次无响应（已等待{elapsed:.1f}秒）。建议：1) 检查ComfyUI是否正常运行；2) 尝试增加超时时间；3) 在ComfyUI界面中检查任务执行情况")
                    time.sleep(poll_interval)
                    continue
                
                consecutive_errors = 0
                
                if prompt_id in history:
                    result = history[prompt_id]
                    print(f"📋 History 数据: {list(result.keys())}")
                    if result.get('outputs'):
                        print(f"✅ 检测到执行完成（尝试{attempt + 1}次）")
                        return result
                    else:
                        # 还在执行中，检查是否有错误
                        if result.get('status'):
                            status = result['status']
                            print(f"📊 状态: {status}")
                            if status.get('completed', False):
                                print(f"✅ 执行完成（状态标记为已完成）")
                                return result
                            elif status.get('error', False):
                                error_msg = status.get('error_message', 'Unknown error')
                                print(f"❌ 执行错误: {error_msg}")
                                raise RuntimeError(f"ComfyUI execution error: {error_msg}")
                else:
                    print(f"⚠️ Prompt ID {prompt_id} 不在 history 中，history keys: {list(history.keys())[:5]}...")
                
                # 每10次尝试打印一次进度
                if attempt % 20 == 0:
                    elapsed = time.time() - start_time
                    print(f"⏳ 轮询中... 尝试{attempt + 1}/{max_attempts}，已等待{elapsed:.1f}秒")
            
            except Exception as e:
                print(f"轮询异常（尝试{attempt + 1}）: {e}")
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors or attempt == max_attempts - 1:
                    raise
                time.sleep(poll_interval)
                continue
            
            time.sleep(poll_interval)
        
        raise TimeoutError(f"ComfyUI执行超时，等待{timeout}秒后仍未完成")

comfyui_manager = ComfyUIManager()

import threading
import time

class SAM3DManager:
    def __init__(self):
        self.body_model = None
        self.body_model_cfg = None
        self.objects_model = None
        self.objects_model_cfg = None
        self.sam3_model = None
        self.sam3_processor = None
        self._body_loaded = False
        self._objects_loaded = False
        self._sam3_loaded = False
        self._last_body_used = None
        self._last_objects_used = None
        self._last_sam3_used = None
        self._unload_timeout = 300  # 5分钟
        self._stop_monitor = False
        self._cached_model_data = None
        self._cached_model_faces = None
        self._cached_model_type = None
        self._cached_multi_person_data = None  # 存储多人数据
        self._cached_selected_persons = None  # 存储选定的人体索引
        self._cached_multi_object_data = None  # 存储多物体数据
        self._cached_selected_objects = None  # 存储选定的物体索引
        self._start_monitor_thread()
    
    def _start_monitor_thread(self):
        def monitor():
            while not self._stop_monitor:
                time.sleep(30)  # 每30秒检查一次
                self._check_and_unload()
        
        self._monitor_thread = threading.Thread(target=monitor, daemon=True)
        self._monitor_thread.start()
    
    def _check_and_unload(self):
        current_time = time.time()
        
        if self._body_loaded and self._last_body_used:
            if current_time - self._last_body_used > self._unload_timeout:
                print("Unloading SAM-3D Body model due to inactivity")
                self._unload_body_model()
        
        if self._objects_loaded and self._last_objects_used:
            if current_time - self._last_objects_used > self._unload_timeout:
                print("Unloading SAM-3D Objects model due to inactivity")
                self._unload_objects_model()
    
    def _unload_body_model(self):
        if self.body_model is not None:
            del self.body_model
            self.body_model = None
        if self.body_model_cfg is not None:
            del self.body_model_cfg
            self.body_model_cfg = None
        self._body_loaded = False
        self._last_body_used = None
        torch.cuda.empty_cache()
    
    def _unload_objects_model(self):
        if self.objects_model is not None:
            del self.objects_model
            self.objects_model = None
        if self.objects_model_cfg is not None:
            del self.objects_model_cfg
            self.objects_model_cfg = None
        self._objects_loaded = False
        self._last_objects_used = None
        torch.cuda.empty_cache()
    
    def _load_body_model(self):
        if self._body_loaded:
            self._last_body_used = time.time()
            return
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading SAM-3D Body model on device: {device}")
        
        try:
            body_model_path = BASE_DIR / "models" / "facebook" / "sam-3d-body-dinov3"
            
            if body_model_path.exists():
                checkpoint_path = body_model_path / "model.ckpt"
                mhr_path = body_model_path / "assets" / "mhr_model.pt"
                
                if checkpoint_path.exists() and mhr_path.exists():
                    sys.path.insert(0, str(body_model_path))
                    try:
                        from sam_3d_body import load_sam_3d_body
                        self.body_model, self.body_model_cfg = load_sam_3d_body(
                            checkpoint_path=str(checkpoint_path),
                            device=device,
                            mhr_path=str(mhr_path)
                        )
                        # 强制使用 float32 精度，避免在某些环境下 BFloat16 导致稀疏矩阵运算报错
                        if self.body_model is not None:
                            self.body_model.float()
                            # 同时强制 backbone 使用 float32，防止模型内部强制转换导致精度不匹配
                            if hasattr(self.body_model, 'backbone_dtype'):
                                self.body_model.backbone_dtype = torch.float32
                        
                        self._body_loaded = True
                        self._last_body_used = time.time()
                        print("SAM-3D Body model loaded successfully")
                    except Exception as e:
                        print(f"Warning: Failed to load SAM-3D Body model: {e}")
                        traceback.print_exc()
                else:
                    print(f"Warning: SAM-3D Body model files not found")
                    print(f"  Checkpoint: {checkpoint_path.exists()}")
                    print(f"  MHR: {mhr_path.exists()}")
        except Exception as e:
            print(f"Warning: Error loading SAM-3D Body model: {e}")
            traceback.print_exc()
    
    def _load_objects_model(self):
        if self._objects_loaded:
            self._last_objects_used = time.time()
            return
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading SAM-3D Objects model on device: {device}")
        
        try:
            objects_model_path = BASE_DIR / "models" / "facebook" / "sam-3d-objects"
            config_path = objects_model_path / "checkpoints" / "pipeline.yaml"
            
            if objects_model_path.exists() and config_path.exists():
                try:
                    try:
                        torch_lib = os.path.join(os.path.dirname(sys.executable), "Lib", "site-packages", "torch", "lib")
                        if os.path.exists(torch_lib):
                            os.add_dll_directory(torch_lib)
                        cuda_paths = [
                            "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.1\\bin",
                            "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\bin"
                        ]
                        for cuda_path in cuda_paths:
                            if os.path.exists(cuda_path):
                                os.add_dll_directory(cuda_path)
                        import pytorch3d
                        from pytorch3d import _C
                        pytorch3d_available = True
                    except ImportError as e:
                        print(f"Warning: pytorch3d not available: {e}")
                        pytorch3d_available = False
                    
                    if pytorch3d_available:
                        kaolin_lib = os.path.join(os.path.dirname(sys.executable), "Lib", "site-packages", "kaolin")
                        if os.path.exists(kaolin_lib):
                            os.add_dll_directory(kaolin_lib)
                        
                        self.objects_model_cfg = OmegaConf.load(config_path)
                        
                        original_cwd = os.getcwd()
                        os.chdir(str(objects_model_path / "checkpoints"))
                        
                        try:
                            from hydra.utils import instantiate
                            self.objects_model = instantiate(self.objects_model_cfg, device=device)
                            if self.objects_model is not None and hasattr(self.objects_model, 'float'):
                                self.objects_model.float()
                            self._objects_loaded = True
                            self._last_objects_used = time.time()
                            print("SAM-3D Objects model loaded successfully")
                        finally:
                            os.chdir(original_cwd)
                    else:
                        print("SAM-3D Objects model skipped due to pytorch3d unavailability")
                except Exception as e:
                    print(f"Warning: Failed to load SAM-3D Objects model: {e}")
                    traceback.print_exc()
            else:
                print("SAM-3D Objects model path not found. Skipping...")
                
        except Exception as e:
            print(f"Warning: Error loading SAM-3D Objects model: {e}")
            traceback.print_exc()
    
    def _load_sam3_model(self):
        """加载SAM3模型用于多人物体检测"""
        if self._sam3_loaded:
            self._last_sam3_used = time.time()
            return
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading SAM3 model on device: {device}")
        
        try:
            sam3_path = BASE_DIR / "sam3"
            
            if sam3_path.exists():
                sys.path.insert(0, str(sam3_path))
                
                try:
                    from sam3.model.sam3_image_processor import Sam3Processor
                    from sam3.model_builder import build_sam3_image_model
                    
                    # 构建SAM3模型
                    self.sam3_model = build_sam3_image_model(
                        bpe_path=None,
                        device=device,
                        eval_mode=True,
                        checkpoint_path=None,
                        load_from_HF=True,
                        enable_segmentation=True,
                        enable_inst_interactivity=True,
                        compile=False,
                    )
                    # 强制使用 float32 精度
                    if self.sam3_model is not None:
                        self.sam3_model.float()
                                            
                    # 创建处理器
                    self.sam3_processor = Sam3Processor(
                        model=self.sam3_model,
                        resolution=1008,
                        device=device,
                        confidence_threshold=0.5
                    )

                    # 创建交互式预测器
                    from sam3.model.sam1_task_predictor import SAM3InteractiveImagePredictor
                    self.sam3_interactive_predictor = SAM3InteractiveImagePredictor(
                        self.sam3_model.inst_interactive_predictor.model
                    )
                    
                    self._sam3_loaded = True
                    self._last_sam3_used = time.time()
                    print("SAM3 model loaded successfully")
                except Exception as e:
                    print(f"Warning: Failed to load SAM3 model: {e}")
                    traceback.print_exc()
            else:
                print(f"Warning: SAM3 model path not found: {sam3_path}")
        except Exception as e:
            print(f"Warning: Error loading SAM3 model: {e}")
            traceback.print_exc()
    
    def _unload_sam3_model(self):
        if self.sam3_model is not None:
            del self.sam3_model
            self.sam3_model = None
        if self.sam3_processor is not None:
            del self.sam3_processor
            self.sam3_processor = None
        self._sam3_loaded = False
        self._last_sam3_used = None
        torch.cuda.empty_cache()
    
    def _array_to_base64(self, arr):
        """将numpy数组（掩码）转换为Base64编码的PNG字符串，带有Alpha通道以支持前端透明绘制"""
        import cv2
        import base64
        import numpy as np
        try:
            # arr 是 (H, W) 的 0-255 掩码
            # 创建 4 通道 BGRA 图像，其中 B,G,R 通道设为 255（白色），A 通道为掩码
            # 这样在前端使用 globalCompositeOperation = 'source-in' 绘制时，只有 Alpha > 0 的区域会被着色
            h, w = arr.shape
            rgba = np.zeros((h, w, 4), dtype=np.uint8)
            rgba[:, :, 0:3] = 255 # BGR = White
            rgba[:, :, 3] = arr   # Alpha = Mask
            _, buffer = cv2.imencode('.png', rgba)
            return base64.b64encode(buffer).decode('utf-8')
        except Exception as e:
            print(f"警告: _array_to_base64 失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def process_image_body(self, image_path):
        # 加载人体模型
        self._load_body_model()
        
        if self.body_model is None or self.body_model_cfg is None:
            raise RuntimeError("SAM-3D Body model is not available. Please check model installation.")
        
        self._last_body_used = time.time()
        
        estimator = SAM3DBodyEstimator(
            sam_3d_body_model=self.body_model,
            model_cfg=self.body_model_cfg,
            human_detector=None,
            human_segmentor=None,
            fov_estimator=None,
        )
        
        # 更新使用时间戳，防止模型在处理过程中被卸载
        self._last_body_used = time.time()
        
        with torch.amp.autocast('cuda', enabled=False):
            outputs = estimator.process_one_image(image_path)
        
        # 再次更新使用时间戳
        self._last_body_used = time.time()
        
        # 确保返回骨骼数据
        for output in outputs:
            # 添加骨骼数据到输出中
            output['pred_joints'] = output.get('pred_keypoints_3d', None)
            output['pred_joint_coords'] = output.get('pred_joint_coords', None)
            output['pred_global_rots'] = output.get('pred_global_rots', None)
            output['body_pose_params'] = output.get('body_pose_params', None)
            output['shape_params'] = output.get('shape_params', None)
            output['scale_params'] = output.get('scale_params', None)
            output['hand_pose_params'] = output.get('hand_pose_params', None)
            output['global_rot'] = output.get('global_rot', None)
            output['expr_params'] = output.get('expr_params', None)
            output['pred_cam_t'] = output.get('pred_cam_t', None)
            output['focal_length'] = output.get('focal_length', None)
        
        return outputs, estimator.faces
    
    def detect_multiple_persons(self, image_path):
        """
        检测图片中的多个人体，返回每个人体的分割掩码和边界框
        使用SAM3模型进行多人物体检测
        """
        self._load_sam3_model()
        self._load_body_model()
        
        if self.sam3_processor is None:
            raise RuntimeError("SAM3 model is not available. Please check model installation.")
        
        if self.body_model is None or self.body_model_cfg is None:
            raise RuntimeError("SAM-3D Body model is not available. Please check model installation.")
        
        self._last_sam3_used = time.time()
        self._last_body_used = time.time()
        
        print(f"=== 检测多人 ===")
        print(f"图片路径: {image_path}")
        
        # 读取图片
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"无法读取图片: {image_path}")
        
        img_height, img_width = img.shape[:2]
        print(f"图片尺寸: {img_width}x{img_height}")
        
        # 使用SAM3模型检测多个人体
        from PIL import Image as PILImage
        
        # 转换为RGB格式
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = PILImage.fromarray(img_rgb)
        
        # 设置图片
        state = self.sam3_processor.set_image(pil_img)
        
        # 使用文本提示检测多个人体
        state = self.sam3_processor.set_text_prompt("person", state)
        
        # 获取检测结果
        boxes = state.get("boxes", None)
        masks = state.get("masks", None)
        scores = state.get("scores", None)
        
        if boxes is None or len(boxes) == 0:
            print("未检测到人体")
            return []
        
        print(f"检测到 {len(boxes)} 个人体")
        
        # 转换为numpy
        boxes_np = boxes.cpu().numpy()
        masks_np = masks.cpu().numpy() if masks is not None else None
        
        # 使用SAM3D Body模型生成3D数据
        # 不再逐个裁剪，而是直接在全图上运行，以保持位置关系
        estimator = SAM3DBodyEstimator(
            sam_3d_body_model=self.body_model,
            model_cfg=self.body_model_cfg,
            human_detector=None,
            human_segmentor=None,
            fov_estimator=None,
        )
        
        print(f"正在全图运行人体估计，保持位置关系...")
        # 显式禁用 autocast 并确保在 float32 下运行，防止 BFloat16 兼容性错误
        with torch.amp.autocast('cuda', enabled=False):
            all_outputs = estimator.process_one_image(img_rgb, bboxes=boxes_np, masks=masks_np)
        
        # 修正多个人体在全局坐标系下的位置偏移
        img_cx, img_cy = img_width / 2.0, img_height / 2.0
        for idx, output in enumerate(all_outputs):
            if 'pred_cam_t' in output and 'focal_length' in output:
                box = boxes_np[idx]
                bbox_cx = (box[0] + box[2]) / 2.0
                bbox_cy = (box[1] + box[3]) / 2.0
                
                # 获取焦距和当前的深度（Z）
                f = float(output['focal_length'])
                tz = float(output['pred_cam_t'][2])
                
                # 计算在全局相机空间中的 XY 偏移补偿
                # 原理：将图像坐标系的偏移转换到相机空间的物理距离
                # 增加 0.55 的调节系数，微调人与人之间的间距感，使布局更紧凑
                dist_scale = 0.55
                tx_offset = (bbox_cx - img_cx) * tz / f * dist_scale
                ty_offset = (bbox_cy - img_cy) * tz / f * dist_scale
                
                # 应用补偿
                output['pred_cam_t'][0] += tx_offset
                output['pred_cam_t'][1] += ty_offset
                
                print(f"人体 {idx} 位置修正: 偏移=[{tx_offset:.3f}, {ty_offset:.3f}], 最终T={output['pred_cam_t']}")
        
        # 整理人体数据
        person_data = []
        for idx, output in enumerate(all_outputs):
            box = boxes_np[idx]
            x_min = max(0, int(box[0]))
            y_min = max(0, int(box[1]))
            x_max = min(img_width, int(box[2]))
            y_max = min(img_height, int(box[3]))
            
            mask = masks_np[idx].squeeze() if masks_np is not None else None
            if mask is not None:
                if mask.shape != (img_height, img_width):
                    mask = cv2.resize(mask.astype(np.uint8), (img_width, img_height), interpolation=cv2.INTER_NEAREST)
                mask = (mask > 0).astype(np.uint8) * 255
            
            # 获取关键点2D位置用于可视化
            keypoints_2d = output.get('pred_keypoints_2d')
            
            person_info = {
                'index': idx,
                'bbox': [x_min, y_min, x_max, y_max],
                'mask_base64': self._array_to_base64(mask) if mask is not None else None,
                'keypoints_2d': keypoints_2d.tolist() if keypoints_2d is not None else None,
                'output': output,
                'score': float(scores[idx]) if scores is not None else 1.0
            }
            
            person_data.append(person_info)
            print(f"人体 {idx}: bbox=[{x_min}, {y_min}, {x_max}, {y_max}], score={person_info['score']:.3f}")
        
        # 缓存多人数据
        self._cached_multi_person_data = person_data
        self._cached_selected_persons = list(range(len(person_data)))  # 默认全选
        
        return person_data
    
    def detect_multiple_objects(self, image_path, prompt="thing"):
        """
        检测图片中的多个物体，返回每个物体的分割掩码和边界框
        使用SAM3模型进行多物体检测（开放词汇分割）
        
        Args:
            image_path: 图片路径
            prompt: 文本提示词，遵循SAM 3官方开放词汇分割方法
        
        注意：此方法仅使用SAM3进行物体检测，不加载sam-3d-objects模型
             sam-3d-objects模型在生成3D模型阶段才加载
        """
        import numpy as np
        
        self._load_sam3_model()
        
        if self.sam3_processor is None:
            raise RuntimeError("SAM3 model is not available. Please check model installation.")
        
        self._last_sam3_used = time.time()
        
        print(f"=== 检测多物体 ===")
        print(f"图片路径: {image_path}")
        
        # 读取图片
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"无法读取图片: {image_path}")
        
        # 保存图片路径供后续生成3D使用
        self._last_image_path = image_path
        
        img_height, img_width = img.shape[:2]
        print(f"图片尺寸: {img_width}x{img_height}")
        
        # 使用SAM3模型检测多个物体
        from PIL import Image as PILImage
        
        # 转换为RGB格式
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = PILImage.fromarray(img_rgb)
        
        # 设置图片
        state = self.sam3_processor.set_image(pil_img)
        
        # 使用文本提示检测多个物体
        # 严格遵循SAM 3官方开放词汇分割方法
        print(f"使用SAM 3文本提示: {prompt}")
        state = self.sam3_processor.set_text_prompt(prompt, state)
        
        # 获取检测结果
        boxes = state.get("boxes", None)
        masks = state.get("masks", None)
        scores = state.get("scores", None)
        
        if boxes is None or len(boxes) == 0:
            print("未检测到物体")
            return []
        
        print(f"检测到 {len(boxes)} 个物体")
        
        # 为每个物体生成3D模型
        object_data = []
        for idx in range(len(boxes)):
            # 获取边界框
            box = boxes[idx]
            # box格式: [x_min, y_min, x_max, y_max] (像素坐标)
            print(f"原始边界框 {idx}: {box}")
            
            # SAM3返回的边界框已经是像素坐标，不需要再乘以图片尺寸
            x_min = max(0, int(box[0].item()))
            y_min = max(0, int(box[1].item()))
            x_max = min(img_width, int(box[2].item()))
            y_max = min(img_height, int(box[3].item()))
            
            print(f"转换后边界框 {idx}: [{x_min}, {y_min}, {x_max}, {y_max}]")
            
            # 获取掩码
            mask = masks[idx].squeeze().cpu().numpy() if masks is not None else None
            if mask is not None:
                print(f"物体 {idx} 原始掩码形状: {mask.shape}, dtype: {mask.dtype}, 值范围: [{mask.min()}, {mask.max()}]")
                # 调整掩码大小到原始图片尺寸
                # 关键修复：bool类型直接转为0-255的uint8，而非先转为0-1再阈值化
                if mask.dtype == bool:
                    mask = mask.astype(np.uint8) * 255
                else:
                    mask = mask.astype(np.uint8)
                    if mask.max() <= 1:  # 归一化的浮点数掩码
                        mask = mask * 255
                mask = cv2.resize(mask, (img_width, img_height), interpolation=cv2.INTER_NEAREST)
                # 确保二值化
                mask = (mask > 128).astype(np.uint8) * 255
                print(f"物体 {idx} 最终掩码形状: {mask.shape}, dtype: {mask.dtype}, 非零像素数: {np.count_nonzero(mask)}")
            else:
                print(f"物体 {idx} 无SAM3掩码，使用边界框创建")
                # 如果没有掩码，使用边界框创建掩码
                mask = np.zeros((img_height, img_width), dtype=np.uint8)
                mask[y_min:y_max, x_min:x_max] = 255
            
            # 确保边界框有效
            if x_max <= x_min or y_max <= y_min:
                print(f"物体 {idx}: 边界框无效，跳过")
                continue
            
            # 保存检测到的物体信息（不含3D数据）
            mask_base64 = self._array_to_base64(mask) if mask is not None else None
            print(f"物体 {idx} mask_base64 生成: {'成功' if mask_base64 else '失败'}")
            object_info = {
                'index': idx,
                'bbox': [x_min, y_min, x_max, y_max],
                'mask_base64': mask_base64,
                'score': float(scores[idx]) if scores is not None else 1.0,
                'cropped_image': None  # 暂不保存裁剪图片
            }
            
            object_data.append(object_info)
            print(f"物体 {idx}: bbox=[{x_min}, {y_min}, {x_max}, {y_max}], score={object_info['score']:.3f}")
        
        # 缓存多物体数据
        self._cached_multi_object_data = object_data
        self._cached_selected_objects = list(range(len(object_data)))  # 默认全选
        
        # 为交互式分割初始化图片（仅当交互式预测器成功初始化且具备有效 model 时）
        try:
            if (hasattr(self, 'sam3_interactive_predictor') and 
                self.sam3_interactive_predictor is not None and
                hasattr(self.sam3_interactive_predictor, 'model') and
                self.sam3_interactive_predictor.model is not None):
                self.sam3_interactive_predictor.set_image(pil_img)
        except Exception as e:
            print(f"警告: 交互式预测器初始化失败，将跳过交互式分割功能: {e}")
        
        return object_data
    
    def interact_object(self, points, labels):
        """
        交互式分割：根据点击点更新掩码
        """
        if self.sam3_interactive_predictor is None:
            raise RuntimeError("SAM3 Interactive Predictor is not available.")
        
        # points: list of [x, y]
        # labels: list of 1 (positive) or 0 (negative)
        
        point_coords = np.array(points)
        point_labels = np.array(labels)
        
        masks, scores, logits = self.sam3_interactive_predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True
        )
        
        # 选择得分最高的掩码
        best_idx = np.argmax(scores)
        mask = masks[best_idx]
        
        # 转换为uint8 255格式
        mask_uint8 = (mask > 0).astype(np.uint8) * 255
        
        # 获取Base64编码
        mask_base64 = self._array_to_base64(mask_uint8)
        
        return mask_base64, float(scores[best_idx])
    
    def process_selected_objects(self, selected_indices=None):
        """
        处理选定的物体，生成3D模型数据
        此时才加载sam-3d-objects模型
        
        Args:
            selected_indices: 选定的物体索引列表，如果为None则使用缓存的选择
        """
        if self._cached_multi_object_data is None or len(self._cached_multi_object_data) == 0:
            raise RuntimeError("没有多物体数据，请先调用 detect_multiple_objects")
        
        if selected_indices is not None:
            self._cached_selected_objects = selected_indices
        
        if self._cached_selected_objects is None or len(self._cached_selected_objects) == 0:
            raise RuntimeError("没有选择任何物体")
        
        print(f"=== 处理选定的物体 ===")
        print(f"选定的物体索引: {self._cached_selected_objects}")
        
        # 加载sam-3d-objects模型（此时才加载）
        self._load_objects_model()
        
        if self.objects_model is None or self.objects_model_cfg is None:
            raise RuntimeError("SAM-3D Objects model is not available. Please check model installation.")
        
        self._last_objects_used = time.time()
        
        # 收集选定的物体数据并生成3D模型
        selected_outputs = []
        for idx in self._cached_selected_objects:
            if idx < len(self._cached_multi_object_data):
                object_data = self._cached_multi_object_data[idx]
                bbox = object_data['bbox']
                
                # 读取原图并裁剪
                image_path = getattr(self, '_last_image_path', None)
                if image_path is None:
                    print(f"警告: 没有图片路径信息")
                    continue
                
                img = cv2.imread(str(image_path))
                if img is None:
                    print(f"无法读取图片")
                    continue
                
                x_min, y_min, x_max, y_max = bbox
                cropped_img = img[y_min:y_max, x_min:x_max]
                
                if cropped_img.size == 0:
                    print(f"物体 {idx}: 裁剪后的图片为空，跳过")
                    continue
                
                # 使用sam-3d-objects模型生成3D数据
                from PIL import Image
                import numpy as np
                import tempfile
                import os
                
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                    temp_image_path = temp_file.name
                    cv2.imwrite(temp_image_path, cropped_img)
                
                try:
                    image = Image.open(temp_image_path).convert('RGBA')
                    image_np = np.array(image)
                    
                    # 使用完整掩码（如果可用）以获得更好的分割效果
                    input_mask = None
                    if object_data.get('mask') is not None:
                        input_mask = np.array(object_data['mask']).astype(np.uint8)
                        # 裁剪掩码
                        input_mask = input_mask[y_min:y_max, x_min:x_max]
                    
                    # 更新使用时间戳，防止模型在处理过程中被卸载
                    self._last_objects_used = time.time()
                    
                    result = self.objects_model.run(
                        image=image_np,
                        mask=input_mask,
                        seed=42,
                        use_vertex_color=True,
                        with_texture_baking=True,
                        with_layout_postprocess=True # 启用布局优化以保持位置一致
                    )
                    
                    # 再次更新使用时间戳
                    self._last_objects_used = time.time()
                    
                    if 'mesh' in result:
                        selected_outputs.append(result)
                        print(f"物体 {idx}: 3D模型生成成功")
                    else:
                        print(f"物体 {idx}: 3D模型生成失败")
                finally:
                    if os.path.exists(temp_image_path):
                        os.unlink(temp_image_path)
        
        print(f"共处理 {len(selected_outputs)} 个物体")
        
        return selected_outputs
    
    def process_selected_persons(self, selected_indices=None):
        """
        处理选定的人体，生成3D模型数据
        
        Args:
            selected_indices: 选定的人体索引列表，如果为None则使用缓存的选择
        """
        if self._cached_multi_person_data is None or len(self._cached_multi_person_data) == 0:
            raise RuntimeError("没有多人数据，请先调用 detect_multiple_persons")
        
        if selected_indices is not None:
            self._cached_selected_persons = selected_indices
        
        if self._cached_selected_persons is None or len(self._cached_selected_persons) == 0:
            raise RuntimeError("没有选择任何人")
        
        print(f"=== 处理选定的人体 ===")
        print(f"选定的人体索引: {self._cached_selected_persons}")
        
        # 获取faces（所有人共享）
        estimator = SAM3DBodyEstimator(
            sam_3d_body_model=self.body_model,
            model_cfg=self.body_model_cfg,
            human_detector=None,
            human_segmentor=None,
            fov_estimator=None,
        )
        faces = estimator.faces
        
        # 收集选定的人体数据
        selected_outputs = []
        for idx in self._cached_selected_persons:
            if idx < len(self._cached_multi_person_data):
                person_data = self._cached_multi_person_data[idx]
                selected_outputs.append(person_data['output'])
                print(f"包含人体 {idx}")
        
        print(f"共处理 {len(selected_outputs)} 个人体")
        
        return selected_outputs, faces
    
    def process_image_objects(self, image_path):
        if self.objects_model is None or self.objects_model_cfg is None:
            raise RuntimeError("SAM-3D Objects model is not available. Please check model installation.")
        
        self._last_objects_used = time.time()
        
        from sam3d_objects.pipeline.inference_pipeline_pointmap import InferencePipelinePointMap
        import torch
        from PIL import Image
        import numpy as np
        
        image = Image.open(image_path).convert('RGBA')
        image_np = np.array(image)
        
        # 更新使用时间戳，防止模型在处理过程中被卸载
        self._last_objects_used = time.time()
        
        result = self.objects_model.run(
            image=image_np,
            mask=None,
            seed=42,
            use_vertex_color=True,
            with_texture_baking=True
        )
        
        # 再次更新使用时间戳
        self._last_objects_used = time.time()
        
        print(f"Result keys: {result.keys()}")
        
        pointmap_colors = result.get('pointmap_colors')
        if pointmap_colors is not None:
            print(f"✓ 检测到pointmap_colors: {pointmap_colors.shape if hasattr(pointmap_colors, 'shape') else type(pointmap_colors)}")
        
        # 辅助函数：检查mesh的纹理信息
        def check_mesh_texture(mesh_obj, idx=0):
            has_uv = False
            has_material = False
            uv_shape = None
            material_type = None
            
            if hasattr(mesh_obj, 'visual'):
                if hasattr(mesh_obj.visual, 'uv') and mesh_obj.visual.uv is not None:
                    has_uv = True
                    uv_shape = mesh_obj.visual.uv.shape
                if hasattr(mesh_obj.visual, 'material') and mesh_obj.visual.material is not None:
                    has_material = True
                    material_type = type(mesh_obj.visual.material).__name__
            
            if has_uv or has_material:
                print(f"✓ 物体 {idx}: 检测到纹理信息 - UV: {has_uv} {uv_shape}, 材质: {has_material} {material_type}")
            return has_uv, has_material
        
        # 优先使用glb字段（包含纹理信息）
        if 'glb' in result and result['glb'] is not None:
            glb_mesh = result['glb']
            print(f"使用glb字段生成mesh")
            print(f"GLB mesh类型: {type(glb_mesh)}")
            
            if isinstance(glb_mesh, list) and len(glb_mesh) > 0:
                mesh = glb_mesh[0]
            else:
                mesh = glb_mesh
            
            if hasattr(mesh, 'vertices') and hasattr(mesh, 'faces'):
                vertices = mesh.vertices
                faces = mesh.faces
                
                # 检查纹理信息
                check_mesh_texture(mesh, 0)
                
                # 调整纹理亮度
                if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'material') and mesh.visual.material is not None:
                    if hasattr(mesh.visual.material, 'baseColorTexture') and mesh.visual.material.baseColorTexture is not None:
                        try:
                            from PIL import ImageEnhance
                            tex = mesh.visual.material.baseColorTexture
                            print(f"  原始纹理尺寸: {tex.size}, 模式: {tex.mode}")
                            
                            # 检查原始亮度
                            tex_array = np.array(tex)
                            print(f"  原始平均像素值: {tex_array.mean(axis=(0,1))}")
                            
                            # 增加亮度和对比度
                            enhancer = ImageEnhance.Brightness(tex)
                            brightened = enhancer.enhance(2.5)  # 2.5倍亮度
                            enhancer = ImageEnhance.Contrast(brightened)
                            mesh.visual.material.baseColorTexture = enhancer.enhance(1.3)  # 1.3倍对比度
                            
                            # 检查修改后的亮度
                            new_tex = mesh.visual.material.baseColorTexture
                            new_tex_array = np.array(new_tex)
                            print(f"  调整后平均像素值: {new_tex_array.mean(axis=(0,1))}")
                            print(f"✓ 已调整纹理亮度")
                        except Exception as e:
                            print(f"⚠ 调整纹理亮度失败: {e}")
                
                # 提取顶点颜色（作为备选）
                vertex_colors = None
                if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'vertex_colors'):
                    vertex_colors = mesh.visual.vertex_colors
                    print(f"✓ 从glb提取顶点颜色: {vertex_colors.shape}")
                
                output_data = [{"pred_vertices": vertices, "vertex_colors": vertex_colors, "mesh": [mesh]}]
                print(f"Vertices shape: {vertices.shape}, Faces shape: {faces.shape}")
                return output_data, faces
        
        # 如果没有glb，使用mesh字段
        if 'mesh' in result:
            mesh = result['mesh']
            print(f"使用mesh字段")
            print(f"Mesh type: {type(mesh)}")
            
            if isinstance(mesh, list):
                if len(mesh) > 0 and hasattr(mesh[0], 'vertices'):
                    vertices = mesh[0].vertices
                    faces = mesh[0].faces
                    vertex_colors = None
                    
                    # 检查纹理信息
                    check_mesh_texture(mesh[0], 0)
                    
                    if hasattr(mesh[0], 'visual'):
                        if hasattr(mesh[0].visual, 'vertex_colors'):
                            vertex_colors = mesh[0].visual.vertex_colors
                            print(f"✓ 从mesh提取顶点颜色: {vertex_colors.shape}")
                        elif hasattr(mesh[0].visual, 'to_color'):
                            try:
                                color_visual = mesh[0].visual.to_color()
                                if hasattr(color_visual, 'vertex_colors'):
                                    vertex_colors = color_visual.vertex_colors
                                    print(f"✓ 从color visual提取顶点颜色: {vertex_colors.shape}")
                            except:
                                pass
                    
                    if vertex_colors is None and pointmap_colors is not None:
                        print(f"✓ 使用pointmap_colors作为顶点颜色")
                        vertex_colors = pointmap_colors
                    
                    output_data = [{"pred_vertices": vertices, "vertex_colors": vertex_colors, "mesh": mesh}]
                else:
                    print(f"Mesh list content: {mesh}")
                    return [], None
            elif hasattr(mesh, 'vertices'):
                vertices = mesh.vertices
                faces = mesh.faces
                vertex_colors = None
                
                # 检查纹理信息
                check_mesh_texture(mesh, 0)
                
                if hasattr(mesh, 'visual'):
                    if hasattr(mesh.visual, 'vertex_colors'):
                        vertex_colors = mesh.visual.vertex_colors
                        print(f"✓ 从mesh提取顶点颜色: {vertex_colors.shape}")
                    elif hasattr(mesh.visual, 'to_color'):
                        try:
                            color_visual = mesh.visual.to_color()
                            if hasattr(color_visual, 'vertex_colors'):
                                vertex_colors = color_visual.vertex_colors
                                print(f"✓ 从color visual提取顶点颜色: {vertex_colors.shape}")
                        except:
                            pass
                
                if vertex_colors is None and pointmap_colors is not None:
                    print(f"✓ 使用pointmap_colors作为顶点颜色")
                    vertex_colors = pointmap_colors
                
                output_data = [{"pred_vertices": vertices, "vertex_colors": vertex_colors, "mesh": [mesh]}]
            else:
                print(f"Unknown mesh structure: {mesh}")
                return [], None
            
            print(f"Vertices shape: {vertices.shape}, Faces shape: {faces.shape}")
            return output_data, faces
        
        print("No mesh in result!")
        return [], None
    
    def _get_rotation_matrix(self, model_type=None):
        """
        根据模型类型获取旋转矩阵
        - body: 180° X轴旋转（使模型直立）
        - objects: 直接使用用户反馈的正确角度（180° X轴）
        """
        mt = model_type or self._cached_model_type or 'body'
        
        if mt == 'objects':
            # 用户反馈：FBX/OBJ需要从90°改为180° X轴
            # 直接使用180° X轴旋转
            rotation_matrix = np.array([
                [1, 0, 0],
                [0, -1, 0],
                [0, 0, -1]
            ])
            print(f"[Rotation] 使用物体模型旋转矩阵 (180° X轴)")
        else:
            # 人体模型继续使用180° X轴旋转
            rotation_matrix = np.array([
                [1, 0, 0],
                [0, -1, 0],
                [0, 0, -1]
            ])
            print(f"[Rotation] 使用人体模型旋转矩阵 (180° X轴)")
        
        return rotation_matrix
    
    def _align_to_ground(self, vertices_np, model_type=None):
        """
        将模型底部对齐到地面（Y=0平面）
        """
        mt = model_type or self._cached_model_type or 'body'

        # 对所有模型类型（body和objects）都应用底部对齐
        y_min = vertices_np[:, 1].min()
        vertices_np = vertices_np.copy()
        vertices_np[:, 1] -= y_min  # 向上偏移使底部接触Y=0平面
        print(f"[Alignment] {mt}模型底部对齐地面，偏移: {-y_min:.2f}")

        return vertices_np
    
    def export_fbx(self, output_data, faces, output_path, model_type=None):
        """
        基于MHR模型的正确FBX导出实现
        
        关键原则：
        1. 使用MHR的127个关节（不是70个）
        2. 直接使用蒙皮后的顶点（pred_vertices）
        3. 使用全局旋转矩阵（pred_global_rots）
        4. 不重新计算蒙皮权重
        """
        try:
            vertices = output_data.get("pred_vertices")

            if vertices is None or faces is None:
                print("Error: No vertices or faces data")
                return False

            import fbx

            vertices_np = vertices.cpu().numpy() if torch.is_tensor(vertices) else vertices
            faces_np = faces.cpu().numpy() if torch.is_tensor(faces) else faces

            # 根据模型类型应用不同的旋转
            mt = model_type or self._cached_model_type or 'body'

            if mt == 'objects':
                # 物体模型：使用180° X轴旋转（与人体模型一致）
                rotation_matrix = np.array([
                    [1, 0, 0],
                    [0, -1, 0],
                    [0, 0, -1]
                ])
                print(f"[Rotation] FBX格式应用物体模型旋转 (180° X轴)")
            else:
                # 人体模型：使用180° X轴旋转
                rotation_matrix = np.array([
                    [1, 0, 0],
                    [0, -1, 0],
                    [0, 0, -1]
                ])
                print(f"[Rotation] FBX格式应用人体模型旋转 (180° X轴)")

            vertices_rotated = vertices_np @ rotation_matrix.T

            vertices_aligned = self._align_to_ground(vertices_rotated)

            vertices_scaled = vertices_aligned * 100.0

            # 创建FBX场景
            manager = fbx.FbxManager.Create()
            scene = fbx.FbxScene.Create(manager, "SAM3D_Scene")
            
            # 创建网格节点
            mesh_node = fbx.FbxNode.Create(scene, "SAM3D_Mesh")
            mesh = fbx.FbxMesh.Create(manager, "SAM3D_Mesh")
            
            mesh_node.SetNodeAttribute(mesh)
            scene.GetRootNode().AddChild(mesh_node)
            
            # 添加顶点
            control_points_count = vertices_scaled.shape[0]
            mesh.InitControlPoints(control_points_count)
            
            for i in range(control_points_count):
                mesh.SetControlPointAt(
                    fbx.FbxVector4(vertices_scaled[i][0], vertices_scaled[i][1], vertices_scaled[i][2]), 
                    i
                )
            
            # 添加面片
            for i in range(faces_np.shape[0]):
                mesh.BeginPolygon()
                for j in range(faces_np.shape[1]):
                    mesh.AddPolygon(faces_np[i][j])
                mesh.EndPolygon()
            
            # 添加材质
            material = fbx.FbxSurfacePhong.Create(manager, "SAM3D_Material")
            material.Diffuse.Set(fbx.FbxDouble3(0.53, 0.53, 0.53))
            material.Specular.Set(fbx.FbxDouble3(0.5, 0.5, 0.5))
            material.Shininess.Set(50.0)
            
            mesh_node.AddMaterial(material)
            
            # 如果是人体模型，添加骨骼
            if mt == 'body':
                print(f"[MHR Export] 开始添加127个MHR关节到FBX场景...")
                self._add_mhr_skeleton_to_fbx(manager, scene, output_data, mesh_node, mesh, vertices_scaled)
                print(f"[MHR Export] 骨骼添加完成")
            else:
                print(f"[MHR Export] 跳过骨骼添加，模型类型: {mt}")
            
            # 配置导出设置
            ios = fbx.FbxIOSettings.Create(manager, fbx.IOSROOT)
            ios.SetBoolProp(fbx.EXP_FBX_EMBEDDED, True)
            ios.SetBoolProp(fbx.EXP_FBX_MATERIAL, True)
            ios.SetBoolProp(fbx.EXP_FBX_TEXTURE, True)
            
            exporter = fbx.FbxExporter.Create(manager, "")
            
            if exporter.Initialize(str(output_path), -1, ios):
                exporter.Export(scene)
                print(f"[MHR Export] ✓ 导出成功: {output_path}")
                print(f"[MHR Export] 文件大小: {output_path.stat().st_size if output_path.exists() else 0} bytes")
                success = True
            else:
                print(f"[MHR Export] ✗ 导出失败: {exporter.GetLastErrorString()}")
                success = False
            
            exporter.Destroy()
            ios.Destroy()
            manager.Destroy()
            
            return success
            
        except Exception as e:
            print(f"[MHR Export] Error exporting FBX: {e}")
            traceback.print_exc()
            return False
    
    def _add_mhr_skeleton_to_fbx(self, manager, scene, output_data, mesh_node, mesh, vertices_scaled):
        """
        基于MHR模型的正确骨骼添加实现
        
        关键原则：
        1. 使用MHR的127个关节（pred_joint_coords）
        2. 使用全局旋转矩阵（pred_global_rots）
        3. 不重新计算蒙皮权重（MHR已经计算好了）
        4. 简化骨骼层级（所有关节作为根的子节点）
        """
        try:
            import fbx
            
            # 获取MHR模型的关节数据
            joint_coords = output_data.get("pred_joint_coords")
            joint_rotations = output_data.get("pred_global_rots")
            
            if joint_coords is None:
                print("[MHR Skeleton] No joint coordinates found, skipping skeleton export")
                return
            
            # 转换为numpy数组
            joint_coords = joint_coords.cpu().numpy() if torch.is_tensor(joint_coords) else joint_coords
            if joint_rotations is not None:
                joint_rotations = joint_rotations.cpu().numpy() if torch.is_tensor(joint_rotations) else joint_rotations
            
            num_joints = joint_coords.shape[0]
            print(f"[MHR Skeleton] 关节数量: {num_joints}")
            
            # 应用相同的旋转和缩放
            mt = self._cached_model_type or 'body'
            if mt == 'objects':
                rotation_matrix = np.array([
                    [1, 0, 0],
                    [0, 0, 1],
                    [0, -1, 0]
                ])
            else:
                rotation_matrix = np.array([
                    [1, 0, 0],
                    [0, -1, 0],
                    [0, 0, -1]
                ])
            
            # 旋转关节坐标
            joint_coords_rotated = joint_coords @ rotation_matrix.T
            
            # 对齐到地面
            joint_coords_aligned = self._align_to_ground(joint_coords_rotated)
            
            # 缩放
            joint_coords_scaled = joint_coords_aligned * 100.0
            
            # 创建根骨骼
            root_bone = fbx.FbxSkeleton.Create(manager, "Root")
            root_bone.SetSkeletonType(fbx.FbxSkeleton.eRoot)
            root_bone_node = fbx.FbxNode.Create(scene, "Root")
            root_bone_node.SetNodeAttribute(root_bone)
            
            # 设置根骨骼位置（使用第一个关节作为根）
            root_pos = joint_coords_scaled[0]
            root_bone_node.LclTranslation.Set(
                fbx.FbxDouble3(root_pos[0], root_pos[1], root_pos[2])
            )
            
            scene.GetRootNode().AddChild(root_bone_node)
            print(f"[MHR Skeleton] 根骨骼创建完成，位置: {root_pos}")
            
            # 创建所有关节骨骼
            bone_nodes = []
            for i in range(num_joints):
                bone_name = f"Joint_{i:03d}"
                bone = fbx.FbxSkeleton.Create(manager, bone_name)
                bone.SetSkeletonType(fbx.FbxSkeleton.eLimbNode)
                bone_node = fbx.FbxNode.Create(scene, bone_name)
                bone_node.SetNodeAttribute(bone)
                
                # 设置骨骼位置
                pos = joint_coords_scaled[i]
                bone_node.LclTranslation.Set(
                    fbx.FbxDouble3(pos[0], pos[1], pos[2])
                )
                
                # 设置骨骼旋转（从旋转矩阵转换为Euler角度）
                if joint_rotations is not None and i < len(joint_rotations):
                    rot_matrix = joint_rotations[i]
                    # 将旋转矩阵转换为Euler角度
                    import math
                    try:
                        # 使用FBX的四元数转换
                        rot_quat = fbx.FbxQuaternion()
                        # 从旋转矩阵创建四元数
                        trace = rot_matrix[0, 0] + rot_matrix[1, 1] + rot_matrix[2, 2]
                        if trace > 0:
                            s = math.sqrt(trace + 1.0) * 2
                            w = 0.25 * s
                            x = (rot_matrix[2, 1] - rot_matrix[1, 2]) / s
                            y = (rot_matrix[0, 2] - rot_matrix[2, 0]) / s
                            z = (rot_matrix[1, 0] - rot_matrix[0, 1]) / s
                        else:
                            if rot_matrix[0, 0] > rot_matrix[1, 1] and rot_matrix[0, 0] > rot_matrix[2, 2]:
                                s = math.sqrt(1.0 + rot_matrix[0, 0] - rot_matrix[1, 1] - rot_matrix[2, 2]) * 2
                                w = (rot_matrix[2, 1] - rot_matrix[1, 2]) / s
                                x = 0.25 * s
                                y = (rot_matrix[0, 1] + rot_matrix[1, 0]) / s
                                z = (rot_matrix[0, 2] + rot_matrix[2, 0]) / s
                            elif rot_matrix[1, 1] > rot_matrix[2, 2]:
                                s = math.sqrt(1.0 + rot_matrix[1, 1] - rot_matrix[0, 0] - rot_matrix[2, 2]) * 2
                                w = (rot_matrix[0, 2] - rot_matrix[2, 0]) / s
                                x = (rot_matrix[0, 1] + rot_matrix[1, 0]) / s
                                y = 0.25 * s
                                z = (rot_matrix[1, 2] + rot_matrix[2, 1]) / s
                            else:
                                s = math.sqrt(1.0 + rot_matrix[2, 2] - rot_matrix[0, 0] - rot_matrix[1, 1]) * 2
                                w = (rot_matrix[1, 0] - rot_matrix[0, 1]) / s
                                x = (rot_matrix[0, 2] + rot_matrix[2, 0]) / s
                                y = (rot_matrix[1, 2] + rot_matrix[2, 1]) / s
                                z = 0.25 * s
                        
                        rot_quat.Set(x, y, z, w)
                        
                        # 将四元数转换为Euler角度
                        euler = rot_quat.DecomposeEuler(fbx.FbxNode.eInheritType.eInheritRSrs)
                        bone_node.LclRotation.Set(fbx.FbxDouble3(euler[0], euler[1], euler[2]))
                    except Exception as e:
                        # 如果转换失败，使用零旋转
                        bone_node.LclRotation.Set(fbx.FbxDouble3(0, 0, 0))
                
                # 将骨骼添加到根节点（简化层级）
                root_bone_node.AddChild(bone_node)
                bone_nodes.append(bone_node)
                
                if (i + 1) % 20 == 0:
                    print(f"[MHR Skeleton] 已创建 {i+1}/{num_joints} 个骨骼...")
            
            print(f"[MHR Skeleton] 所有{num_joints}个骨骼创建完成")
            
            # 注意：不添加蒙皮权重
            # 原因：MHR模型已经计算了蒙皮，pred_vertices已经是蒙皮后的顶点
            # 重新计算蒙皮权重会导致错误
            # 如果需要在3D软件中调整姿势，需要使用MHR的内部蒙皮权重（目前未暴露）
            
            print(f"[MHR Skeleton] 骨骼系统添加完成（{num_joints}个关节）")
            
        except Exception as e:
            print(f"[MHR Skeleton] Error adding skeleton to FBX: {e}")
            traceback.print_exc()
    
    def export_glb(self, output_data, faces, output_path, model_type=None):
        try:
            vertices = output_data.get("pred_vertices")
            vertex_colors = output_data.get("vertex_colors")
            
            # 检查是否有mesh字段（物体模型）
            mesh_list = output_data.get("mesh", [])
            has_mesh_texture = False
            source_mesh = None
            
            if isinstance(mesh_list, list) and len(mesh_list) > 0:
                source_mesh = mesh_list[0]
                if hasattr(source_mesh, 'visual') and hasattr(source_mesh.visual, 'material'):
                    if source_mesh.visual.material is not None and hasattr(source_mesh.visual.material, 'baseColorTexture'):
                        if source_mesh.visual.material.baseColorTexture is not None:
                            has_mesh_texture = True
                            print(f"[export_glb] 检测到mesh纹理，将优先使用")

            if vertices is not None and faces is not None:
                import trimesh

                vertices_np = vertices.cpu().numpy() if torch.is_tensor(vertices) else vertices
                faces_np = faces.cpu().numpy() if torch.is_tensor(faces) else faces

                # 根据模型类型应用不同的旋转
                mt = model_type or self._cached_model_type or 'body'

                if mt == 'objects':
                    # 物体模型：使用180° X轴旋转（与人体模型一致）
                    rotation_matrix = np.array([
                        [1, 0, 0],
                        [0, -1, 0],
                        [0, 0, -1]
                    ])
                    print(f"[Rotation] GLB格式应用物体模型旋转 (180° X轴)")
                else:
                    # 人体模型：使用180° X轴旋转
                    rotation_matrix = np.array([
                        [1, 0, 0],
                        [0, -1, 0],
                        [0, 0, -1]
                    ])
                    print(f"[Rotation] GLB格式应用人体模型旋转 (180° X轴)")

                vertices_rotated = vertices_np @ rotation_matrix.T

                # 应用预测的相机平移（pred_cam_t），以保持多个人体/物体的相对位置关系
                pred_cam_t = output_data.get("pred_cam_t")
                if pred_cam_t is not None:
                    pred_cam_t_np = pred_cam_t.cpu().numpy() if torch.is_tensor(pred_cam_t) else pred_cam_t
                    # 确保平移向量形状正确 (3,) 或 (1, 3)
                    pred_cam_t_np = pred_cam_t_np.flatten()
                    if pred_cam_t_np.shape[0] == 3:
                        # 转换平移到旋转后的坐标系
                        # 注意：如果旋转是 180度X轴，平移也需要相应处理
                        t_rotated = pred_cam_t_np @ rotation_matrix.T
                        vertices_rotated += t_rotated
                        print(f"[Alignment] 应用相机平移: {t_rotated}")

                # 注意：移除单个人体的底部对齐，以免破坏多个人体间的相对垂直位置
                # vertices_aligned = self._align_to_ground(vertices_rotated)
                vertices_aligned = vertices_rotated
                
                # 如果有mesh纹理，使用source_mesh并更新顶点
                if has_mesh_texture and source_mesh is not None:
                    print(f"[export_glb] 使用mesh纹理导出")
                    
                    # 检查并调整纹理亮度
                    try:
                        from PIL import ImageEnhance
                        tex = source_mesh.visual.material.baseColorTexture
                        print(f"  原始纹理尺寸: {tex.size}, 模式: {tex.mode}")
                        
                        # 检查原始亮度
                        tex_array = np.array(tex)
                        print(f"  原始平均像素值: {tex_array.mean(axis=(0,1))}")
                        
                        # 增加亮度和对比度
                        enhancer = ImageEnhance.Brightness(tex)
                        brightened = enhancer.enhance(2.5)  # 2.5倍亮度
                        enhancer = ImageEnhance.Contrast(brightened)
                        source_mesh.visual.material.baseColorTexture = enhancer.enhance(1.3)  # 1.3倍对比度
                        
                        # 检查修改后的亮度
                        new_tex = source_mesh.visual.material.baseColorTexture
                        new_tex_array = np.array(new_tex)
                        print(f"  调整后平均像素值: {new_tex_array.mean(axis=(0,1))}")
                        print(f"✓ 已调整纹理亮度")
                    except Exception as e:
                        print(f"⚠ 调整纹理亮度失败: {e}")
                    
                    # 直接修改source_mesh的顶点和面
                    source_mesh.vertices = vertices_aligned
                    source_mesh.faces = faces_np
                    mesh = source_mesh
                    print(f"✓ 已应用mesh纹理")
                else:
                    # 没有mesh纹理，创建新mesh
                    mesh = trimesh.Trimesh(vertices=vertices_aligned, faces=faces_np)
                    
                    if vertex_colors is not None:
                        vertex_colors_np = vertex_colors.cpu().numpy() if torch.is_tensor(vertex_colors) else vertex_colors
                        if len(vertex_colors_np.shape) == 2 and vertex_colors_np.shape[1] >= 3:
                            mesh.visual.vertex_colors = vertex_colors_np
                            print(f"✓ 应用顶点颜色: {vertex_colors_np.shape}")
                        else:
                            print(f"⚠ 顶点颜色格式错误: {vertex_colors_np.shape}")
                            mesh.visual = trimesh.visual.ColorVisuals(
                                mesh,
                                face_colors=np.full((len(faces_np), 4), [136, 136, 136, 255], dtype=np.uint8)
                            )
                    else:
                        print(f"⚠ 无顶点颜色，使用默认灰色")
                        mesh.visual = trimesh.visual.ColorVisuals(
                            mesh,
                            face_colors=np.full((len(faces_np), 4), [136, 136, 136, 255], dtype=np.uint8)
                        )
                
                mesh.export(str(output_path))
                print(f"Exported GLB: {output_path}")
                return True

            return False
        except Exception as e:
            print(f"Error exporting GLB: {e}")
            traceback.print_exc()
            return False
    
    def export_obj(self, output_data, faces, output_path, model_type=None):
        try:
            vertices = output_data.get("pred_vertices")

            if vertices is not None and faces is not None:
                import trimesh

                vertices_np = vertices.cpu().numpy() if torch.is_tensor(vertices) else vertices
                faces_np = faces.cpu().numpy() if torch.is_tensor(faces) else faces

                # 根据模型类型应用不同的旋转
                mt = model_type or self._cached_model_type or 'body'

                if mt == 'objects':
                    # 物体模型：使用180° X轴旋转（与人体模型一致）
                    rotation_matrix = np.array([
                        [1, 0, 0],
                        [0, -1, 0],
                        [0, 0, -1]
                    ])
                    print(f"[Rotation] OBJ格式应用物体模型旋转 (180° X轴)")
                else:
                    # 人体模型：使用180° X轴旋转
                    rotation_matrix = np.array([
                        [1, 0, 0],
                        [0, -1, 0],
                        [0, 0, -1]
                    ])
                    print(f"[Rotation] OBJ格式应用人体模型旋转 (180° X轴)")
                vertices_rotated = vertices_np @ rotation_matrix.T
                print(f"[Rotation] OBJ格式应用180° X轴旋转")
                
                vertices_aligned = self._align_to_ground(vertices_rotated)
                
                mesh = trimesh.Trimesh(vertices=vertices_aligned, faces=faces_np)
                
                mesh.visual = trimesh.visual.ColorVisuals(
                    mesh,
                    face_colors=np.full((len(faces_np), 4), [136, 136, 136, 255], dtype=np.uint8)
                )
                
                mesh.export(str(output_path))
                print(f"Exported OBJ: {output_path}")
                return True
            
            return False
        except Exception as e:
            print(f"Error exporting OBJ: {e}")
            traceback.print_exc()
            return False
    
    def export_ply(self, output_data, faces, output_path, model_type=None):
        try:
            vertices = output_data.get("pred_vertices")

            if vertices is not None and faces is not None:
                import trimesh

                vertices_np = vertices.cpu().numpy() if torch.is_tensor(vertices) else vertices
                faces_np = faces.cpu().numpy() if torch.is_tensor(faces) else faces

                # 根据模型类型应用不同的旋转
                mt = model_type or self._cached_model_type or 'body'

                if mt == 'objects':
                    # 物体模型：使用180° X轴旋转（与人体模型一致）
                    rotation_matrix = np.array([
                        [1, 0, 0],
                        [0, -1, 0],
                        [0, 0, -1]
                    ])
                    print(f"[Rotation] PLY格式应用物体模型旋转 (180° X轴)")
                else:
                    # 人体模型：使用180° X轴旋转
                    rotation_matrix = np.array([
                        [1, 0, 0],
                        [0, -1, 0],
                        [0, 0, -1]
                    ])
                    print(f"[Rotation] PLY格式应用人体模型旋转 (180° X轴)")
                vertices_rotated = vertices_np @ rotation_matrix.T

                # 物体模型PLY：底部对齐到Y=0
                if mt == 'objects':
                    y_min = vertices_rotated[:, 1].min()
                    y_max = vertices_rotated[:, 1].max()
                    height = y_max - y_min
                    vertices_aligned = vertices_rotated.copy()
                    vertices_aligned[:, 1] -= y_min  # 对齐底部到Y=0
                    print(f"[Alignment] PLY物体模型对齐：高度={height:.2f}")
                else:
                    vertices_aligned = self._align_to_ground(vertices_rotated)

                mesh = trimesh.Trimesh(vertices=vertices_aligned, faces=faces_np)
                
                mesh.visual = trimesh.visual.ColorVisuals(
                    mesh,
                    face_colors=np.full((len(faces_np), 4), [136, 136, 136, 255], dtype=np.uint8)
                )
                
                mesh.export(str(output_path))
                print(f"Exported PLY: {output_path}")
                return True
            
            return False
        except Exception as e:
            print(f"Error exporting PLY: {e}")
            traceback.print_exc()
            return False
    
    def export_stl(self, output_data, faces, output_path, model_type=None):
        try:
            vertices = output_data.get("pred_vertices")

            if vertices is not None and faces is not None:
                import trimesh

                vertices_np = vertices.cpu().numpy() if torch.is_tensor(vertices) else vertices
                faces_np = faces.cpu().numpy() if torch.is_tensor(faces) else faces

                # 根据模型类型应用不同的旋转
                mt = model_type or self._cached_model_type or 'body'

                if mt == 'objects':
                    # 物体模型：使用180° X轴旋转（与人体模型一致）
                    rotation_matrix = np.array([
                        [1, 0, 0],
                        [0, -1, 0],
                        [0, 0, -1]
                    ])
                    print(f"[Rotation] STL格式应用物体模型旋转 (180° X轴)")
                else:
                    # 人体模型：使用180° X轴旋转
                    rotation_matrix = np.array([
                        [1, 0, 0],
                        [0, -1, 0],
                        [0, 0, -1]
                    ])
                    print(f"[Rotation] STL格式应用人体模型旋转 (180° X轴)")
                vertices_rotated = vertices_np @ rotation_matrix.T

                # 物体模型STL：底部对齐到Y=0
                if mt == 'objects':
                    y_min = vertices_rotated[:, 1].min()
                    y_max = vertices_rotated[:, 1].max()
                    height = y_max - y_min
                    vertices_aligned = vertices_rotated.copy()
                    vertices_aligned[:, 1] -= y_min  # 对齐底部到Y=0
                    print(f"[Alignment] STL物体模型对齐：高度={height:.2f}")
                else:
                    vertices_aligned = self._align_to_ground(vertices_rotated)

                mesh = trimesh.Trimesh(vertices=vertices_aligned, faces=faces_np)
                
                mesh.visual = trimesh.visual.ColorVisuals(
                    mesh,
                    face_colors=np.full((len(faces_np), 4), [136, 136, 136, 255], dtype=np.uint8)
                )
                
                mesh.export(str(output_path))
                print(f"Exported STL: {output_path}")
                return True
            
            return False
        except Exception as e:
            print(f"Error exporting STL: {e}")
            traceback.print_exc()
            return False
    
    def export_mhr_json(self, output_data, faces, output_path, model_type=None):
        """
        导出MHR JSON格式（官方支持）
        
        MHR格式包含：
        - 网格顶点和面片
        - 骨骼关节和旋转
        - 身体姿态参数
        - 形状和缩放参数
        - 手部姿态参数
        - 面部表情参数
        - 相机参数
        """
        try:
            vertices = output_data.get("pred_vertices")

            if vertices is None or faces is None:
                print("[MHR Export] Error: No vertices or faces data")
                return False

            vertices_np = vertices.cpu().numpy() if torch.is_tensor(vertices) else vertices
            faces_np = faces.cpu().numpy() if torch.is_tensor(faces) else faces

            # 根据模型类型应用不同的旋转
            mt = model_type or self._cached_model_type or 'body'

            if mt == 'objects':
                # 物体模型：使用-90° X轴旋转
                rotation_matrix = np.array([
                    [1, 0, 0],
                    [0, 0, 1],
                    [0, -1, 0]
                ])
                print(f"[Rotation] MHR格式应用物体模型旋转 (-90° X轴)")
            else:
                # 人体模型：使用180° X轴旋转
                rotation_matrix = np.array([
                    [1, 0, 0],
                    [0, -1, 0],
                    [0, 0, -1]
                ])
                print(f"[Rotation] MHR格式应用人体模型旋转 (180° X轴)")

            vertices_rotated = vertices_np @ rotation_matrix.T

            vertices_aligned = self._align_to_ground(vertices_rotated)

            # 准备MHR数据
            mhr_data = {}

            # 网格数据
            mhr_data["mesh"] = {
                "vertices": vertices_aligned.tolist(),
                "faces": faces_np.tolist(),
            }

            # 关键点数据
            if "pred_keypoints_3d" in output_data:
                keypoints_3d = output_data["pred_keypoints_3d"]
                keypoints_3d_np = keypoints_3d.cpu().numpy() if torch.is_tensor(keypoints_3d) else keypoints_3d
                mhr_data["keypoints_3d"] = keypoints_3d_np.tolist()

            if "pred_keypoints_2d" in output_data:
                keypoints_2d = output_data["pred_keypoints_2d"]
                keypoints_2d_np = keypoints_2d.cpu().numpy() if torch.is_tensor(keypoints_2d) else keypoints_2d
                mhr_data["keypoints_2d"] = keypoints_2d_np.tolist()

            # 关节坐标
            if "pred_joint_coords" in output_data:
                joint_coords = output_data["pred_joint_coords"]
                joint_coords_np = joint_coords.cpu().numpy() if torch.is_tensor(joint_coords) else joint_coords
                # 应用相同的旋转和对齐
                joint_coords_rotated = joint_coords_np @ rotation_matrix.T
                joint_coords_aligned = self._align_to_ground(joint_coords_rotated)
                mhr_data["joint_coords"] = joint_coords_aligned.tolist()

            # 骨骼数据
            if "joint_global_rots" in output_data:
                joint_rots = output_data["joint_global_rots"]
                joint_rots_np = joint_rots.cpu().numpy() if torch.is_tensor(joint_rots) else joint_rots
                mhr_data["skeleton"] = {
                    "joint_global_rots": joint_rots_np.tolist(),
                }
                if "global_rot" in output_data:
                    global_rot = output_data["global_rot"]
                    global_rot_np = global_rot.cpu().numpy() if torch.is_tensor(global_rot) else global_rot
                    mhr_data["skeleton"]["global_rot"] = global_rot_np.tolist()

            # 姿态参数
            if "body_pose" in output_data:
                body_pose = output_data["body_pose"]
                body_pose_np = body_pose.cpu().numpy() if torch.is_tensor(body_pose) else body_pose
                mhr_data["pose_parameters"] = {
                    "body_pose": body_pose_np.tolist(),
                }

            # 手部姿态
            if "hand" in output_data:
                hand = output_data["hand"]
                hand_np = hand.cpu().numpy() if torch.is_tensor(hand) else hand
                if hand_np.size > 27:
                    mhr_data["pose_parameters"]["hand_pose"] = {
                        "left": hand_np[:27].tolist(),
                        "right": hand_np[27:].tolist(),
                    }
                else:
                    mhr_data["pose_parameters"]["hand_pose"] = {
                        "left": hand_np.tolist(),
                        "right": [],
                    }

            # 面部表情
            if "face" in output_data:
                face = output_data["face"]
                face_np = face.cpu().numpy() if torch.is_tensor(face) else face
                mhr_data["pose_parameters"]["face"] = face_np.tolist()

            # 身体参数
            if "shape" in output_data:
                shape = output_data["shape"]
                shape_np = shape.cpu().numpy() if torch.is_tensor(shape) else shape
                mhr_data["body_parameters"] = {
                    "shape": shape_np.tolist(),
                }

            if "scale" in output_data:
                scale = output_data["scale"]
                scale_np = scale.cpu().numpy() if torch.is_tensor(scale) else scale
                mhr_data["body_parameters"]["scale"] = scale_np.tolist()

            # 相机参数
            if "pred_cam_t" in output_data:
                cam_t = output_data["pred_cam_t"]
                cam_t_np = cam_t.cpu().numpy() if torch.is_tensor(cam_t) else cam_t
                mhr_data["camera"] = {
                    "translation": cam_t_np.tolist(),
                }

            if "focal_length" in output_data:
                focal_length = output_data["focal_length"]
                focal_length_np = focal_length.cpu().numpy() if torch.is_tensor(focal_length) else focal_length
                mhr_data["camera"]["focal_length"] = float(focal_length_np)

            # 元数据
            mhr_data["metadata"] = {
                "format": "MHR",
                "version": "1.0",
                "model": "SAM-3D-Body",
                "num_vertices": int(vertices_aligned.shape[0]),
                "num_faces": int(faces_np.shape[0]),
            }

            # 保存JSON文件
            output_path_str = str(output_path).replace('.fbx', '.mhr.json')
            with open(output_path_str, 'w') as f:
                json.dump(mhr_data, f, indent=2)

            print(f"[MHR Export] ✓ 导出成功: {output_path_str}")
            print(f"[MHR Export] 文件大小: {Path(output_path_str).stat().st_size} bytes")
            return True

        except Exception as e:
            print(f"[MHR Export] Error exporting MHR JSON: {e}")
            traceback.print_exc()
            return False

    def export_mhr_json_multi(self, outputs, faces, output_path):
        """
        导出多人MHR JSON格式（合并）
        """
        try:
            combined_data = {
                "num_people": len(outputs),
                "people": [],
            }

            for idx, output_data in enumerate(outputs):
                person_data = {
                    "id": idx,
                }

                # 网格数据
                vertices = output_data.get("pred_vertices")
                vertices_np = vertices.cpu().numpy() if torch.is_tensor(vertices) else vertices
                faces_np = faces.cpu().numpy() if torch.is_tensor(faces) else faces

                # 应用旋转和对齐
                rotation_matrix = np.array([
                    [1, 0, 0],
                    [0, -1, 0],
                    [0, 0, -1]
                ])
                vertices_rotated = vertices_np @ rotation_matrix.T
                vertices_aligned = self._align_to_ground(vertices_rotated)

                person_data["mesh"] = {
                    "vertices": vertices_aligned.tolist(),
                    "faces": faces_np.tolist(),
                }

                # 关键点数据
                if "pred_keypoints_3d" in output_data:
                    keypoints_3d = output_data["pred_keypoints_3d"]
                    keypoints_3d_np = keypoints_3d.cpu().numpy() if torch.is_tensor(keypoints_3d) else keypoints_3d
                    person_data["keypoints_3d"] = keypoints_3d_np.tolist()

                if "pred_keypoints_2d" in output_data:
                    keypoints_2d = output_data["pred_keypoints_2d"]
                    keypoints_2d_np = keypoints_2d.cpu().numpy() if torch.is_tensor(keypoints_2d) else keypoints_2d
                    person_data["keypoints_2d"] = keypoints_2d_np.tolist()

                # 关节坐标
                if "pred_joint_coords" in output_data:
                    joint_coords = output_data["pred_joint_coords"]
                    joint_coords_np = joint_coords.cpu().numpy() if torch.is_tensor(joint_coords) else joint_coords
                    joint_coords_rotated = joint_coords_np @ rotation_matrix.T
                    joint_coords_aligned = self._align_to_ground(joint_coords_rotated)
                    person_data["joint_coords"] = joint_coords_aligned.tolist()

                # 骨骼数据
                if "joint_global_rots" in output_data:
                    joint_rots = output_data["joint_global_rots"]
                    joint_rots_np = joint_rots.cpu().numpy() if torch.is_tensor(joint_rots) else joint_rots
                    person_data["skeleton"] = {
                        "joint_global_rots": joint_rots_np.tolist(),
                    }
                    if "global_rot" in output_data:
                        global_rot = output_data["global_rot"]
                        global_rot_np = global_rot.cpu().numpy() if torch.is_tensor(global_rot) else global_rot
                        person_data["skeleton"]["global_rot"] = global_rot_np.tolist()

                combined_data["people"].append(person_data)

            # 元数据
            combined_data["metadata"] = {
                "format": "MHR-Combined",
                "version": "1.0",
                "model": "SAM-3D-Body",
            }

            # 保存JSON文件
            output_path_str = str(output_path).replace('.fbx', '.mhr.json')
            with open(output_path_str, 'w') as f:
                json.dump(combined_data, f, indent=2)

            print(f"[MHR Export] ✓ 导出成功（多人）: {output_path_str}")
            print(f"[MHR Export] 文件大小: {Path(output_path_str).stat().st_size} bytes")
            return True

        except Exception as e:
            print(f"[MHR Export] Error exporting multi-person MHR JSON: {e}")
            traceback.print_exc()
            return False

    def _convert_ply_to_fbx(self, ply_path, fbx_path):
        try:
            import open3d as o3d
            mesh = o3d.io.read_triangle_mesh(str(ply_path))
            o3d.io.write_triangle_mesh(str(fbx_path), mesh)
            Path(ply_path).unlink()
            print(f"Exported FBX: {fbx_path}")
        except Exception as e:
            print(f"Could not convert PLY to FBX: {e}")
            traceback.print_exc()
    
    def export_fbx_multi(self, meshes, output_path):
        """
        导出多个mesh到单个FBX文件，包含骨骼
        
        基于MHR模型的正确实现
        """
        try:
            import fbx
            import numpy as np
            
            manager = fbx.FbxManager.Create()
            scene = fbx.FbxScene.Create(manager, "SAM3D_Multi-Person_Scene")
            
            # 检查是否有缓存的多人数据
            if not isinstance(self._cached_model_data, list):
                print("[MHR Multi Export] Warning: No multi-person data available for skeleton export")
                return self._export_meshes_only(meshes, output_path, manager, scene)
            
            # 为每个人体创建mesh和骨骼
            for idx, mesh in enumerate(meshes):
                node = fbx.FbxNode.Create(scene, f"Person_{idx}")
                fbx_mesh = fbx.FbxMesh.Create(manager, f"Mesh_{idx}")
                
                node.SetNodeAttribute(fbx_mesh)
                scene.GetRootNode().AddChild(node)
                
                vertices = mesh.vertices
                faces = mesh.faces
                
                # 应用旋转
                mt = self._cached_model_type or 'body'
                if mt == 'objects':
                    rotation_matrix = np.array([
                        [1, 0, 0],
                        [0, 0, 1],
                        [0, -1, 0]
                    ])
                else:
                    rotation_matrix = np.array([
                        [1, 0, 0],
                        [0, -1, 0],
                        [0, 0, -1]
                    ])
                
                vertices_rotated = vertices @ rotation_matrix.T
                vertices_aligned = self._align_to_ground(vertices_rotated)
                vertices_scaled = vertices_aligned * 100.0
                
                # 添加顶点
                control_points_count = vertices_scaled.shape[0]
                fbx_mesh.InitControlPoints(control_points_count)
                
                for i in range(control_points_count):
                    fbx_mesh.SetControlPointAt(
                        fbx.FbxVector4(vertices_scaled[i][0], vertices_scaled[i][1], vertices_scaled[i][2]), 
                        i
                    )
                
                # 添加面片
                for i in range(faces.shape[0]):
                    fbx_mesh.BeginPolygon()
                    for j in range(faces.shape[1]):
                        fbx_mesh.AddPolygon(faces[i][j])
                    fbx_mesh.EndPolygon()
                
                # 添加骨骼
                if idx < len(self._cached_model_data):
                    output_data = self._cached_model_data[idx]
                    print(f"[MHR Multi Export] 为人体 {idx} 添加骨骼...")
                    self._add_mhr_skeleton_to_fbx(manager, scene, output_data, node, fbx_mesh, vertices_scaled)
            
            # 配置导出设置
            ios = fbx.FbxIOSettings.Create(manager, fbx.IOSROOT)
            ios.SetBoolProp(fbx.EXP_FBX_EMBEDDED, True)
            ios.SetBoolProp(fbx.EXP_FBX_MATERIAL, True)
            ios.SetBoolProp(fbx.EXP_FBX_TEXTURE, True)
            
            exporter = fbx.FbxExporter.Create(manager, "")
            
            if exporter.Initialize(str(output_path), -1, ios):
                exporter.Export(scene)
                print(f"[MHR Multi Export] ✓ 导出成功: {output_path}")
                print(f"[MHR Multi Export] 文件大小: {output_path.stat().st_size if output_path.exists() else 0} bytes")
                success = True
            else:
                print(f"[MHR Multi Export] ✗ 导出失败: {exporter.GetLastErrorString()}")
                success = False
            
            exporter.Destroy()
            ios.Destroy()
            manager.Destroy()
            
            return success
        except Exception as e:
            print(f"[MHR Multi Export] Error exporting multi-person FBX: {e}")
            traceback.print_exc()
            return False
    
    def _export_meshes_only(self, meshes, output_path, manager, scene):
        """
        只导出mesh，不添加骨骼（用于兼容性）
        """
        try:
            import fbx
            import numpy as np
            
            for idx, mesh in enumerate(meshes):
                node = fbx.FbxNode.Create(scene, f"Person_{idx}")
                fbx_mesh = fbx.FbxMesh.Create(manager, f"Mesh_{idx}")
                
                node.SetNodeAttribute(fbx_mesh)
                scene.GetRootNode().AddChild(node)
                
                vertices = mesh.vertices
                faces = mesh.faces
                
                # 应用旋转
                mt = self._cached_model_type or 'body'
                if mt == 'objects':
                    rotation_matrix = np.array([
                        [1, 0, 0],
                        [0, 0, 1],
                        [0, -1, 0]
                    ])
                else:
                    rotation_matrix = np.array([
                        [1, 0, 0],
                        [0, -1, 0],
                        [0, 0, -1]
                    ])
                
                vertices_rotated = vertices @ rotation_matrix.T
                vertices_aligned = self._align_to_ground(vertices_rotated)
                vertices_scaled = vertices_aligned * 100.0
                
                # 添加顶点
                control_points_count = vertices_scaled.shape[0]
                fbx_mesh.InitControlPoints(control_points_count)
                
                for i in range(control_points_count):
                    fbx_mesh.SetControlPointAt(
                        fbx.FbxVector4(vertices_scaled[i][0], vertices_scaled[i][1], vertices_scaled[i][2]), 
                        i
                    )
                
                # 添加面片
                for i in range(faces.shape[0]):
                    fbx_mesh.BeginPolygon()
                    for j in range(faces.shape[1]):
                        fbx_mesh.AddPolygon(faces[i][j])
                    fbx_mesh.EndPolygon()
            
            # 配置导出设置
            ios = fbx.FbxIOSettings.Create(manager, fbx.IOSROOT)
            ios.SetBoolProp(fbx.EXP_FBX_EMBEDDED, True)
            ios.SetBoolProp(fbx.EXP_FBX_MATERIAL, True)
            ios.SetBoolProp(fbx.EXP_FBX_TEXTURE, True)
            
            exporter = fbx.FbxExporter.Create(manager, "")
            
            if exporter.Initialize(str(output_path), -1, ios):
                exporter.Export(scene)
                print(f"[MHR Export] ✓ 导出成功（仅mesh）: {output_path}")
                print(f"[MHR Export] 文件大小: {output_path.stat().st_size if output_path.exists() else 0} bytes")
                success = True
            else:
                print(f"[MHR Export] ✗ 导出失败: {exporter.GetLastErrorString()}")
                success = False
            
            exporter.Destroy()
            ios.Destroy()
            
            return success
        except Exception as e:
            print(f"[MHR Export] Error exporting meshes only: {e}")
            traceback.print_exc()
            return False

sam3d_manager = SAM3DManager()

@app.route('/api/status', methods=['GET'])
def get_status():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cuda_available = torch.cuda.is_available()

    body_model_path = BASE_DIR / "models" / "facebook" / "sam-3d-body-dinov3"
    objects_model_path = BASE_DIR / "models" / "facebook" / "sam-3d-objects"

    sam3d_body_available = body_model_path.exists() and (body_model_path / "model.ckpt").exists()
    sam3d_objects_available = objects_model_path.exists() and (objects_model_path / "checkpoints" / "pipeline.yaml").exists()

    # 检查ComfyUI连接
    comfyui_available = False
    try:
        import requests
        response = requests.get(f"{comfyui_manager.base_url}/system_stats", timeout=2)
        comfyui_available = response.status_code == 200
    except:
        comfyui_available = False

    return jsonify({
        "backend": True,
        "cuda": cuda_available,
        "sam3d_body": sam3d_body_available,
        "sam3d_objects": sam3d_objects_available,
        "comfyui": comfyui_available
    })

@app.route('/api/llm/models', methods=['GET'])
def get_llm_models():
    provider = request.args.get('provider', 'ollama')
    base_url = request.args.get('base_url', 'http://localhost:11434/v1')
    api_key = request.args.get('api_key')
    
    models = llm_manager.get_models(provider, base_url, api_key)
    return jsonify({"models": models})

@app.route('/api/llm/chat', methods=['POST'])
def llm_chat():
    data = request.json
    provider = data.get('provider', 'ollama')
    base_url = data.get('base_url', 'http://localhost:11434/v1')
    model = data.get('model', 'llama2')
    messages = data.get('messages', [])
    api_key = data.get('api_key')
    
    def generate():
        for chunk in llm_manager.stream_chat(provider, base_url, model, messages, api_key):
            yield f"data: {json.dumps({'content': chunk})}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/api/generate/prompt', methods=['POST'])
def generate_prompt():
    data = request.json
    user_input = data.get('user_input', '')
    prompt_type = data.get('type', 'text_to_image')

    # 优化的中文prompt，要求生成符合SAM-3D规范的提示词
    system_prompt = """你是专业的AI图像提示词生成专家。你的任务是生成符合SAM-3D Body和SAM-3D Objects图生3D最佳实践的中文图像生成提示词。

核心原则：
1. 完整主体原则：如果描述的是人，必须是完整全身（从头到脚），绝对禁止只生成半身（膝盖以上、半身、面部特写等）；如果描述的是物品（如汽车），必须是完整物品（完整的车身、可见所有主要部件）
2. 最佳视角原则：必须使用透视角度，立体感最强的拍摄角度，禁止平视、俯视、仰视等扁平角度
3. 3D重建适配：确保视角适合3D重建，包含足够的深度信息

具体要求：
4. 必须输出完整的一个自然段，绝对禁止分段
5. 绝对禁止任何解释、说明、问候语、结束语等一切多余废话
6. 只输出最终的提示词内容，不要有任何额外文字
7. 图片要求：透视角度（略微低角度仰视或四分之三侧视），清晰的轮廓和边缘，充足且均匀的光照，简洁的中性背景（浅灰或白色），展现真实的材质和纹理细节，适合3D重建的高质量照片风格
8. 避免使用"白模"、"线框"等词，要生成真实材质的图片
9. 提示词要详细但简洁，约100-150字
10. 绝对禁止在结尾添加固定短语如"全体车身，整体。"，提示词应该自然流畅地结束

人体描述时的要求：
- 主体要完整，从头到脚都要可见
- 禁止"半身"、"上半身"、"特写"等描述
- 姿态要自然站立或完整动作

物品描述时的要求（重要）：
- 主体要完整，展示主要轮廓和整体结构
- 禁止只展示局部的描述
- 不要在结尾添加固定短语
- 必须强调背景简洁：纯色背景、摄影棚背景、无多余装饰、无复杂纹理背景、无场景元素
- 物体要位于画面中央，背景干净无干扰，类似电商产品摄影风格"""

    user_prompt = f"基于以下描述，生成符合SAM-3D图生3D最佳实践的高质量图像提示词（注意：主体必须完整，使用透视角度增强立体感）：{user_input}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    provider = data.get('provider', 'ollama')
    base_url = data.get('base_url', 'http://localhost:11434/v1')
    model = data.get('model', 'llama2')
    api_key = data.get('api_key')
    
    print(f"Generating prompt with provider={provider}, base_url={base_url}, model={model}")
    
    try:
        full_response = ""
        for chunk in llm_manager.stream_chat(provider, base_url, model, messages, api_key):
            full_response += chunk
        
        print(f"Generated prompt: {full_response[:100]}...")
        
        return jsonify({
            'success': True,
            'prompt': full_response
        })
    except Exception as e:
        error_msg = str(e)
        print(f"Error generating prompt: {error_msg}")
        traceback.print_exc()
        
        if "connection" in error_msg.lower() or "refused" in error_msg.lower():
            return jsonify({
                'success': False,
                'error': f'无法连接到LLM服务 ({provider})，请检查：\n1. 服务是否启动\n2. URL是否正确: {base_url}\n3. 防火墙是否阻止连接'
            })
        else:
            return jsonify({
                'success': False,
                'error': f'生成提示词失败: {error_msg}'
            })

@app.route('/api/image/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image file provided'})
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No image file selected'})
    
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        filename = f"uploaded_{int(time.time())}.png"
        filepath = OUTPUT_DIR / filename
        file.save(str(filepath))
        
        return jsonify({
            'success': True,
            'image_path': str(filepath),
            'image_url': f"/api/file/{filename}"
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/image/generate', methods=['POST'])
def generate_image():
    data = request.json
    prompt = data.get('prompt', '')
    negative_prompt = data.get('negative_prompt', '')
    workflow_name = data.get('workflow', 'flux2_klein_9b')
    resolution = data.get('resolution', '1024x1024')
    seed = data.get('seed')

    # 如果用户没有指定种子，生成一个随机种子确保每次生成不同的图片
    if seed is None:
        seed = random.randint(0, 2147483647)
        print(f"🔀 自动生成随机种子: {seed}")
    else:
        print(f"🎯 使用用户指定种子: {seed}")

    print(f"=== 开始生成图片 ===")
    print(f"Prompt: {prompt[:100] if len(prompt) > 100 else prompt}")
    print(f"Workflow: {workflow_name}")
    print(f"Resolution: {resolution}")
    print(f"Negative prompt: {negative_prompt[:50] if negative_prompt else 'None'}")
    print(f"Seed: {seed}")

    width, height = RESOLUTIONS.get(resolution, (1024, 1024))

    # 为每次请求生成新的client_id，避免WebSocket连接冲突
    client_id = str(uuid.uuid4())
    print(f"Client ID: {client_id}")

    try:
        workflow = comfyui_manager.load_workflow(workflow_name)
        workflow = comfyui_manager.modify_workflow(
            workflow, prompt, negative_prompt, width, height, seed
        )

        result = comfyui_manager.queue_prompt(workflow, client_id=client_id)
        prompt_id = result.get('prompt_id')

        print(f"Prompt ID: {prompt_id}")
        print(f"Queue result: {result}")

        if prompt_id:
            print(f"等待ComfyUI完成...")

            # 直接使用HTTP轮询方式等待完成
            print(f"等待ComfyUI执行完成...")
            history = comfyui_manager.wait_for_completion(prompt_id, client_id=client_id)
            print(f"✅ ComfyUI执行完成")

            outputs = history.get('outputs', {})
            print(f"Outputs keys: {list(outputs.keys())}")

            outputs = history.get('outputs', {})
            print(f"Outputs keys: {list(outputs.keys())}")

            for node_id, node_data in outputs.items():
                if 'images' in node_data:
                    for img in node_data['images']:
                        filename = img['filename']
                        print(f"图片文件名: {filename}")

                        # 尝试多个可能的目录
                        possible_paths = [
                            COMFYUI_OUTPUT_DIR / filename,  # 配置的目录
                            BASE_DIR.parent / "ComfyUI_windows_portable" / "ComfyUI" / "output" / filename,  # 父目录
                            Path("D:/ComfyUI_windows_portable/ComfyUI/output") / filename,  # 硬编码路径
                        ]

                        filepath = None
                        for path in possible_paths:
                            print(f"检查路径: {path}")
                            if path.exists():
                                filepath = path
                                print(f"✅ 找到图片文件: {filepath}")
                                break

                        if filepath:
                            file_size = filepath.stat().st_size
                            print(f"图片文件大小: {file_size} bytes")
                            print(f"图片URL: /api/comfyui/image/{filename}")
                            print(f"图片目录: {filepath.parent}")

                            return jsonify({
                                'success': True,
                                'image_path': str(filepath),
                                'image_url': f"/api/comfyui/image/{filename}",
                                'filename': filename,
                                'prompt_id': prompt_id
                            })
                        else:
                            print(f"⚠️ 警告: 图片文件不存在于任何可能的路径")
                            print(f"尝试的路径:")
                            for path in possible_paths:
                                print(f"  - {path} (存在: {path.exists()})")

                            return jsonify({
                                'success': False,
                                'error': f'图片文件不存在: {filename}。请检查ComfyUI输出目录'
                            })

        return jsonify({'success': False, 'error': 'Failed to generate image'})
    except Exception as e:
        # 在抛出异常之前，尝试检查是否有新生成的图片（即使轮询失败）
        print(f"❌ 生成图片异常: {e}")
        print(f"❌ 尝试查找可能已生成的图片...")

        try:
            # 查找最近30秒内修改的图片文件
            current_time = time.time()
            for directory in [
                COMFYUI_OUTPUT_DIR,
                BASE_DIR.parent / "ComfyUI_windows_portable" / "ComfyUI" / "output",
                Path("D:/ComfyUI_windows_portable/ComfyUI/output"),
            ]:
                if directory.exists():
                    for filepath in directory.glob("*.png"):
                        file_mtime = filepath.stat().st_mtime
                        if current_time - file_mtime < 30:  # 30秒内修改的文件
                            print(f"✅ 找到可能的新图片: {filepath}")
                            return jsonify({
                                'success': True,
                                'image_path': str(filepath),
                                'image_url': f"/api/comfyui/image/{filepath.name}",
                                'filename': filepath.name,
                                'prompt_id': prompt_id if 'prompt_id' in locals() else 'unknown',
                                'warning': f'图片生成过程中检测到异常: {str(e)}。但已找到新生成的图片。'
                            })
        except Exception as find_error:
            print(f"⚠️ 查找图片时出错: {find_error}")

        print(f"❌ 未找到新生成的图片")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/image/edit', methods=['POST'])
def edit_image():
    data = request.json
    image_path = data.get('image_path', '')
    prompt = data.get('prompt', '')
    negative_prompt = data.get('negative_prompt', '')
    workflow_name = data.get('workflow', 'flux2_klein_edit')
    resolution = data.get('resolution', '1024x1024')
    seed = data.get('seed')
    
    width, height = RESOLUTIONS.get(resolution, (1024, 1024))

    # 为每次请求生成新的client_id，避免WebSocket连接冲突
    client_id = str(uuid.uuid4())

    try:
        upload_result = comfyui_manager.upload_image(image_path)
        image_name = upload_result.get('name', Path(image_path).name)

        workflow = comfyui_manager.load_workflow(workflow_name)
        workflow = comfyui_manager.modify_workflow(
            workflow, prompt, negative_prompt, width, height, seed, image_name
        )

        result = comfyui_manager.queue_prompt(workflow, client_id=client_id)
        prompt_id = result.get('prompt_id')

        if prompt_id:
            history = comfyui_manager.wait_for_completion(prompt_id, client_id=client_id)
            
            for node_id, node_data in history.get('outputs', {}).items():
                if 'images' in node_data:
                    for img in node_data['images']:
                        filename = img['filename']
                        return jsonify({
                            'success': True,
                            'filename': filename,
                            'prompt_id': prompt_id
                        })
        
        return jsonify({'success': False, 'error': 'Failed to edit image'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/3d/detect-persons', methods=['POST'])
def detect_persons():
    """
    检测图片中的多个人体
    """
    data = request.json
    image_path = data.get('image_path', '')
    
    print(f"=== 检测多人请求 ===")
    print(f"图片路径: {image_path}")
    print(f"图片路径类型: {type(image_path)}")
    
    # 检查文件是否存在
    from pathlib import Path
    path_obj = Path(image_path)
    print(f"文件存在: {path_obj.exists()}")
    print(f"文件是绝对路径: {path_obj.is_absolute()}")
    
    try:
        sam3d_manager._load_body_model()
        person_data = sam3d_manager.detect_multiple_persons(image_path)
        
        print(f"检测结果: {len(person_data)} 个人体")
        
        if person_data and len(person_data) > 0:
            # 读取图片用于返回
            import base64
            with open(image_path, 'rb') as f:
                img_data = f.read()
                img_base64 = base64.b64encode(img_data).decode('utf-8')
            
            # 只返回基本信息，不包含 output 字段（因为 output 包含 numpy ndarray）
            persons_summary = []
            for person in person_data:
                persons_summary.append({
                    'index': person['index'],
                    'bbox': person['bbox'],
                    'mask_base64': person['mask_base64'],
                    'score': person['score']
                })
            
            return jsonify({
                'success': True,
                'persons': persons_summary,
                'image_base64': img_base64,
                'total_count': len(person_data)
            })
        else:
            print("返回错误: 未检测到人体")
            return jsonify({
                'success': False,
                'error': '未检测到人体'
            })
    except Exception as e:
        print(f"Error detecting persons: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/3d/generate-multi', methods=['POST'])
def generate_3d_multi():
    """
    生成选定的人体的3D模型
    """
    data = request.json
    selected_indices = data.get('selected_indices', [])
    
    print(f"=== 生成多人3D模型 ===")
    print(f"选定的人体索引: {selected_indices}")
    
    try:
        # 处理选定的人体
        output_data, faces = sam3d_manager.process_selected_persons(selected_indices)
        
        if output_data and len(output_data) > 0 and faces is not None:
            # 缓存多人数据
            sam3d_manager._cached_model_data = output_data
            sam3d_manager._cached_model_faces = faces
            sam3d_manager._cached_model_type = 'body'
            
            print(f"3D模型生成成功！共 {len(output_data)} 个人体")
            
            return jsonify({
                'success': True,
                'model_type': 'body',
                'person_count': len(output_data)
            })
        else:
            return jsonify({
                'success': False,
                'error': '生成3D模型失败'
            })
    except Exception as e:
        print(f"Error generating multi-person 3D model: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/3d/detect-objects', methods=['POST'])
def detect_objects():
    """
    检测图片中的多个物体
    严格遵循SAM 3官方方法：使用文本提示进行开放词汇分割
    """
    data = request.json
    image_path = data.get('image_path', '')
    prompt = data.get('prompt', 'thing')
    
    print(f"=== 检测多物体请求 ===")
    print(f"图片路径: {image_path}")
    print(f"提示词: {prompt}")
    print(f"图片路径类型: {type(image_path)}")
    
    # 检查文件是否存在
    from pathlib import Path
    path_obj = Path(image_path)
    print(f"文件存在: {path_obj.exists()}")
    print(f"文件是绝对路径: {path_obj.is_absolute()}")
    
    try:
        # 传递用户输入的提示词
        object_data = sam3d_manager.detect_multiple_objects(image_path, prompt)
        
        print(f"检测结果: {len(object_data)} 个物体")
        
        if object_data and len(object_data) > 0:
            # 读取图片用于返回
            import base64
            with open(image_path, 'rb') as f:
                img_data = f.read()
                img_base64 = base64.b64encode(img_data).decode('utf-8')
            
            # 只返回基本信息，不包含 result 字段（因为 result 包含复杂对象）
            objects_summary = []
            for obj in object_data:
                objects_summary.append({
                    'index': obj['index'],
                    'bbox': obj['bbox'],
                    'mask_base64': obj['mask_base64'],
                    'score': obj['score']
                })
            
            return jsonify({
                'success': True,
                'objects': objects_summary,
                'image_base64': img_base64,
                'total_count': len(object_data)
            })
        else:
            print("返回错误: 未检测到物体")
            return jsonify({
                'success': False,
                'error': '未检测到物体'
            })
    except Exception as e:
        print(f"Error detecting objects: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/3d/generate-multi-objects', methods=['POST'])
def generate_3d_multi_objects():
    """
    生成选定的物体的3D模型
    """
    data = request.json
    selected_indices = data.get('selected_indices', [])
    
    print(f"=== 生成多物体3D模型 ===")
    print(f"选定的物体索引: {selected_indices}")
    
    try:
        # 处理选定的物体
        selected_outputs = sam3d_manager.process_selected_objects(selected_indices)
        
        if selected_outputs and len(selected_outputs) > 0:
            # 缓存多物体数据
            sam3d_manager._cached_model_data = selected_outputs
            sam3d_manager._cached_model_type = 'objects'
            
            print(f"3D模型生成成功！共 {len(selected_outputs)} 个物体")
            
            return jsonify({
                'success': True,
                'model_type': 'objects',
                'object_count': len(selected_outputs)
            })
        else:
            return jsonify({
                'success': False,
                'error': '生成3D模型失败'
            })
    except Exception as e:
        print(f"Error generating multi-object 3D model: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/3d/generate', methods=['POST'])
def generate_3d():
    data = request.json
    image_path = data.get('image_path', '')
    model_type = data.get('model_type', 'body')
    
    print(f"=== 3D Generation Request ===")
    print(f"Image path: {image_path}")
    print(f"Model type: {model_type}")
    print(f"============================")
    
    try:
        if model_type == 'body':
            print("Loading and processing with Body model...")
            sam3d_manager._load_body_model()
            output_data, faces = sam3d_manager.process_image_body(image_path)
        elif model_type == 'objects':
            print("Loading and processing with Objects model...")
            sam3d_manager._load_objects_model()
            output_data, faces = sam3d_manager.process_image_objects(image_path)
        else:
            return jsonify({'success': False, 'error': 'Invalid model type. Choose body or objects.'})
        
        if output_data and len(output_data) > 0 and faces is not None:
            sam3d_manager._cached_model_data = output_data[0]
            sam3d_manager._cached_model_faces = faces
            sam3d_manager._cached_model_type = model_type
            
            print(f"3D model generated successfully! Type: {model_type}")
            print(f"Vertices: {output_data[0].get('pred_vertices').shape if hasattr(output_data[0].get('pred_vertices'), 'shape') else 'N/A'}")
            print(f"Faces: {faces.shape if hasattr(faces, 'shape') else 'N/A'}")
            
            return jsonify({
                'success': True,
                'model_type': model_type
            })
        else:
            print(f"Failed to generate 3D model. Output data: {len(output_data) if output_data else 0}, Faces: {faces is not None}")
            return jsonify({'success': False, 'error': 'Failed to generate 3D model'})
    except Exception as e:
        print(f"Error generating 3D model: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/3d/export', methods=['POST'])
def export_3d():
    import sys
    from pathlib import Path
    
    # 获取请求数据
    data = request.json
    format_type = data.get('format', 'fbx')
    model_type = sam3d_manager._cached_model_type or 'body'
    
    # 打开日志文件（使用绝对路径）
    log_path = Path('d:/good/sam-3d/server_export.log').resolve()
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write("\n" + "="*60 + "\n")
        f.write("EXPORT_3D函数被调用\n")
        f.write("="*60 + "\n")
        
        f.write(f"format_type={format_type}, model_type={model_type}\n")
        f.write(f"_cached_model_data type={type(sam3d_manager._cached_model_data)}\n")
        f.write(f"_cached_model_data keys={sam3d_manager._cached_model_data.keys() if isinstance(sam3d_manager._cached_model_data, dict) else 'N/A'}\n")
        
        # 检查模型数据
        if sam3d_manager._cached_model_data is None:
            f.write("ERROR: No model data available\n")
            return jsonify({'success': False, 'error': 'No model data available. Please generate a model first.'})
        
        # 对于objects模型，数据格式不同
        # _cached_model_data是列表，每个元素是字典，包含mesh等字段
        is_objects_model = model_type == 'objects' and isinstance(sam3d_manager._cached_model_data, list) and len(sam3d_manager._cached_model_data) > 0 and isinstance(sam3d_manager._cached_model_data[0], dict) and 'mesh' in sam3d_manager._cached_model_data[0]
        
        f.write(f"is_objects_model={is_objects_model}\n")
        f.write(f"  - model_type == 'objects': {model_type == 'objects'}\n")
        f.write(f"  - isinstance(_cached_model_data, list): {isinstance(sam3d_manager._cached_model_data, list)}\n")
        if isinstance(sam3d_manager._cached_model_data, list) and len(sam3d_manager._cached_model_data) > 0:
            f.write(f"  - _cached_model_data[0] type: {type(sam3d_manager._cached_model_data[0])}\n")
            if isinstance(sam3d_manager._cached_model_data[0], dict):
                f.write(f"  - 'mesh' in _cached_model_data[0]: {'mesh' in sam3d_manager._cached_model_data[0]}\n")
                f.write(f"  - _cached_model_data[0] keys: {list(sam3d_manager._cached_model_data[0].keys())}\n")
        f.flush()
    
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        existing_files = list(OUTPUT_DIR.glob(f"*.{format_type}"))
        max_number = 0
        for file in existing_files:
            try:
                number = int(file.stem)
                if number > max_number:
                    max_number = number
            except ValueError:
                pass
        
        next_number = max_number + 1
        output_name = f"{next_number:04d}"
        output_path = OUTPUT_DIR / f"{output_name}.{format_type}"
        
        # 处理objects模型导出
        print(f"DEBUG: is_objects_model={is_objects_model}, model_type={model_type}")
        print(f"DEBUG: _cached_model_data type={type(sam3d_manager._cached_model_data)}")
        if is_objects_model:
            print(f"=== 导出物体3D模型 ===")
            print(f"物体数量: {len(sam3d_manager._cached_model_data)}")
            print(f"格式: {format_type}")

            import trimesh
            import io

            # 物体模型：根据用户反馈，使用 -90° X轴旋转（绕X轴逆时针90度）
            # 旋转矩阵：绕X轴旋转 -90度
            rotation_matrix = np.array([
                [1, 0, 0],
                [0, 0, 1],
                [0, -1, 0]
            ])

            # 处理所有物体
            meshes = []
            for obj_idx, result in enumerate(sam3d_manager._cached_model_data):
                mesh_list = result.get('mesh', [])

                if isinstance(mesh_list, list) and len(mesh_list) > 0:
                    mesh = mesh_list[0]

                    if hasattr(mesh, 'vertices'):
                        vertices_np = mesh.vertices
                        if torch.is_tensor(vertices_np):
                            vertices_np = vertices_np.cpu().numpy()

                        # 应用旋转
                        vertices_rotated = vertices_np @ rotation_matrix.T

                        if hasattr(mesh, 'faces'):
                            faces_np = mesh.faces
                            if torch.is_tensor(faces_np):
                                faces_np = faces_np.cpu().numpy()
                        else:
                            faces_np = np.array([])

                        # 对齐到地面
                        y_min = vertices_rotated[:, 1].min()
                        vertices_offset = vertices_rotated.copy()
                        vertices_offset[:, 1] -= y_min

                        # 为每个物体添加水平偏移，避免重叠
                        offset_x = obj_idx * 3.0  # 每个物体间隔3个单位
                        vertices_offset[:, 0] += offset_x

                        # 检查原始mesh是否有纹理
                        has_texture = False
                        uv_coords = None

                        if hasattr(mesh, 'visual'):
                            if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
                                uv_coords = mesh.visual.uv
                                has_texture = True
                                print(f"✓ 物体 {obj_idx}: 检测到UV坐标 {uv_coords.shape}")
                            if hasattr(mesh.visual, 'material') and mesh.visual.material is not None:
                                has_texture = True
                                print(f"✓ 物体 {obj_idx}: 检测到材质 {type(mesh.visual.material)}")

                        # 创建trimesh mesh
                        if has_texture and uv_coords is not None:
                            # 有纹理，尝试保留
                            try:
                                # 调整纹理亮度
                                if mesh.visual.material is not None and hasattr(mesh.visual.material, 'baseColorTexture') and mesh.visual.material.baseColorTexture is not None:
                                    from PIL import ImageEnhance
                                    tex = mesh.visual.material.baseColorTexture
                                    enhancer = ImageEnhance.Brightness(tex)
                                    brightened = enhancer.enhance(2.5)
                                    enhancer = ImageEnhance.Contrast(brightened)
                                    adjusted_tex = enhancer.enhance(1.3)
                                    mesh.visual.material.baseColorTexture = adjusted_tex
                                    print(f"✓ 物体 {obj_idx}: 已调整纹理亮度")

                                # 直接修改mesh的顶点和面
                                mesh.vertices = vertices_offset
                                mesh.faces = faces_np
                                export_mesh = mesh
                                print(f"✓ 物体 {obj_idx}: 已应用纹理")
                            except Exception as e:
                                print(f"⚠ 物体 {obj_idx}: 纹理处理失败，使用普通mesh: {e}")
                                export_mesh = trimesh.Trimesh(vertices=vertices_offset, faces=faces_np)
                        else:
                            # 没有纹理，创建普通mesh
                            export_mesh = trimesh.Trimesh(vertices=vertices_offset, faces=faces_np)

                            # 尝试复制顶点颜色
                            if hasattr(mesh, 'visual'):
                                try:
                                    if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
                                        export_mesh.visual.vertex_colors = mesh.visual.vertex_colors
                                        print(f"✓ 物体 {obj_idx}: 已应用顶点颜色")
                                except Exception as e:
                                    print(f"⚠ 物体 {obj_idx}: 复制顶点颜色失败: {e}")

                        meshes.append(export_mesh)
                        print(f"✓ 物体 {obj_idx}: 顶点数 {len(vertices_np)}, 面数 {len(faces_np)}")

            if len(meshes) == 0:
                return jsonify({'success': False, 'error': 'No valid mesh data found'})

            print(f"共处理 {len(meshes)} 个物体")

            # 导出所有物体
            if format_type in ['glb', 'obj', 'ply', 'stl']:
                # 合并所有mesh并导出
                combined_mesh = trimesh.util.concatenate(meshes)
                combined_mesh.export(str(output_path))
                print(f"Exported {format_type.upper()}: {output_path}")
                success = True
            elif format_type == 'fbx':
                # 使用trimesh导出为临时GLB，然后转换为FBX
                # 或者使用export_fbx_multi方法
                temp_glb_path = output_path.with_suffix('.temp.glb')
                combined_mesh = trimesh.util.concatenate(meshes)
                combined_mesh.export(str(temp_glb_path))

                # 读取GLB并转换为FBX格式
                try:
                    import fbx
                    manager = fbx.FbxManager.Create()
                    scene = fbx.FbxScene.Create(manager, "SAM3D_Objects_Scene")

                    # 为每个mesh创建FBX节点
                    for mesh_idx, mesh in enumerate(meshes):
                        mesh_node = fbx.FbxNode.Create(scene, f"Object_{mesh_idx}")
                        fbx_mesh = fbx.FbxMesh.Create(manager, f"Object_{mesh_idx}_Mesh")
                        mesh_node.SetNodeAttribute(fbx_mesh)
                        scene.GetRootNode().AddChild(mesh_node)

                        vertices = mesh.vertices
                        faces = mesh.faces

                        # 应用缩放（与export_fbx一致）
                        vertices_scaled = vertices * 100.0

                        # 添加顶点
                        control_points_count = vertices_scaled.shape[0]
                        fbx_mesh.InitControlPoints(control_points_count)
                        for i in range(control_points_count):
                            fbx_mesh.SetControlPointAt(
                                fbx.FbxVector4(vertices_scaled[i][0], vertices_scaled[i][1], vertices_scaled[i][2]),
                                i
                            )

                        # 添加面片
                        for i in range(faces.shape[0]):
                            fbx_mesh.BeginPolygon()
                            for j in range(faces.shape[1]):
                                fbx_mesh.AddPolygon(faces[i][j])
                            fbx_mesh.EndPolygon()

                        # 添加材质
                        material = fbx.FbxSurfacePhong.Create(manager, f"Object_{mesh_idx}_Material")
                        material.Diffuse.Set(fbx.FbxDouble3(0.53, 0.53, 0.53))
                        material.Specular.Set(fbx.FbxDouble3(0.5, 0.5, 0.5))
                        material.Shininess.Set(50.0)
                        mesh_node.AddMaterial(material)

                    # 配置导出设置
                    ios = fbx.FbxIOSettings.Create(manager, fbx.IOSROOT)
                    ios.SetBoolProp(fbx.EXP_FBX_EMBEDDED, True)
                    ios.SetBoolProp(fbx.EXP_FBX_MATERIAL, True)
                    ios.SetBoolProp(fbx.EXP_FBX_TEXTURE, True)

                    exporter = fbx.FbxExporter.Create(manager, "")
                    exporter.Initialize(str(output_path), -1, ios)
                    exporter.Export(scene)
                    exporter.Destroy()
                    scene.Destroy()
                    manager.Destroy()

                    # 删除临时文件
                    if temp_glb_path.exists():
                        temp_glb_path.unlink()

                    print(f"Exported FBX: {output_path}")
                    success = True
                except Exception as e:
                    import traceback
                    print(f"FBX export error: {e}")
                    traceback.print_exc()
                    # 如果FBX导出失败，返回GLB
                    if temp_glb_path.exists():
                        glb_path = output_path.with_suffix('.glb')
                        temp_glb_path.rename(glb_path)
                        return jsonify({'success': True, 'filename': glb_path.name, 'format': 'glb', 'note': 'FBX export failed, exported as GLB instead'})
                    success = False
            else:
                return jsonify({'success': False, 'error': f'Unsupported format for objects: {format_type}. Use glb, obj, ply, stl, or fbx.'})
        
        # 处理人体模型导出（原有逻辑）
        elif isinstance(sam3d_manager._cached_model_data, list) and len(sam3d_manager._cached_model_data) > 1:
            print(f"=== 导出多人3D模型 ===")
            print(f"人体数量: {len(sam3d_manager._cached_model_data)}")
            print(f"格式: {format_type}")
            
            if format_type == 'mhr':
                success = sam3d_manager.export_mhr_json_multi(sam3d_manager._cached_model_data, sam3d_manager._cached_model_faces, output_path)
            elif format_type in ['glb', 'obj', 'ply', 'stl']:
                import trimesh
                meshes = []
                for idx, output_data in enumerate(sam3d_manager._cached_model_data):
                    vertices = output_data.get("pred_vertices")
                    faces = sam3d_manager._cached_model_faces

                    vertices_np = vertices.cpu().numpy() if torch.is_tensor(vertices) else vertices
                    faces_np = faces.cpu().numpy() if torch.is_tensor(faces) else faces

                    rotation_matrix = np.array([
                        [1, 0, 0],
                        [0, -1, 0],
                        [0, 0, -1]
                    ])
                    
                    vertices_rotated = vertices_np @ rotation_matrix.T
                    
                    y_min = vertices_rotated[:, 1].min()
                    vertices_aligned = vertices_rotated.copy()
                    vertices_aligned[:, 1] -= y_min
                    
                    offset_x = idx * 2.0
                    vertices_aligned[:, 0] += offset_x
                    
                    mesh = trimesh.Trimesh(vertices=vertices_aligned, faces=faces_np)
                    meshes.append(mesh)
                
                combined_mesh = trimesh.util.concatenate(meshes)
                combined_mesh.export(str(output_path))
                print(f"Exported {format_type.upper()}: {output_path}")
                success = True
            elif format_type == 'fbx':
                import trimesh
                meshes = []
                for idx, output_data in enumerate(sam3d_manager._cached_model_data):
                    vertices = output_data.get("pred_vertices")
                    faces = sam3d_manager._cached_model_faces
                    vertices_np = vertices.cpu().numpy() if torch.is_tensor(vertices) else vertices
                    faces_np = faces.cpu().numpy() if torch.is_tensor(faces) else faces
                    mesh = trimesh.Trimesh(vertices=vertices_np, faces=faces_np)
                    meshes.append(mesh)
                success = sam3d_manager.export_fbx_multi(meshes, output_path)
            else:
                return jsonify({'success': False, 'error': f'Unsupported format: {format_type}'})
        else:
            # 单人模式
            if format_type == 'mhr':
                success = sam3d_manager.export_mhr_json(sam3d_manager._cached_model_data, sam3d_manager._cached_model_faces, output_path, sam3d_manager._cached_model_type)
            elif format_type == 'fbx':
                success = sam3d_manager.export_fbx(sam3d_manager._cached_model_data, sam3d_manager._cached_model_faces, output_path, sam3d_manager._cached_model_type)
            elif format_type == 'glb':
                success = sam3d_manager.export_glb(sam3d_manager._cached_model_data, sam3d_manager._cached_model_faces, output_path, sam3d_manager._cached_model_type)
            elif format_type == 'obj':
                success = sam3d_manager.export_obj(sam3d_manager._cached_model_data, sam3d_manager._cached_model_faces, output_path, sam3d_manager._cached_model_type)
            elif format_type == 'ply':
                success = sam3d_manager.export_ply(sam3d_manager._cached_model_data, sam3d_manager._cached_model_faces, output_path, sam3d_manager._cached_model_type)
            elif format_type == 'stl':
                success = sam3d_manager.export_stl(sam3d_manager._cached_model_data, sam3d_manager._cached_model_faces, output_path, sam3d_manager._cached_model_type)
            else:
                return jsonify({'success': False, 'error': f'Unsupported format: {format_type}'})
        
        if success:
            return jsonify({
                'success': True,
                'filename': f"{output_name}.{format_type}",
                'format': format_type
            })
        else:
            return jsonify({'success': False, 'error': f'Failed to export {format_type}'})
    except Exception as e:
        import traceback
        print(f"ERROR in export_3d: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/objects/interact', methods=['POST'])
def interact_object():
    """交互式物体分割"""
    try:
        data = request.json
        points = data.get('points', [])
        labels = data.get('labels', [])
        
        if not points:
            return jsonify({'success': False, 'error': 'No points provided'})
            
        mask_base64, score = sam3d_manager.interact_object(points, labels)
        
        return jsonify({
            'success': True,
            'mask_base64': mask_base64,
            'score': score
        })
    except Exception as e:
        print(f"Error in interact_object: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/3d/model', methods=['GET'])
def get_cached_model():
    import numpy as np
    
    # 对于objects模型，数据格式不同，不检查 _cached_model_faces
    model_type = sam3d_manager._cached_model_type or 'body'
    if model_type == 'objects':
        if sam3d_manager._cached_model_data is None:
            return jsonify({'success': False, 'error': 'No model data available'})
    else:
        if sam3d_manager._cached_model_data is None or sam3d_manager._cached_model_faces is None:
            return jsonify({'success': False, 'error': 'No model data available'})

    try:
        if model_type == 'objects':
            print(f"=== 返回物体3D模型 ===")
            print(f"_cached_model_data type: {type(sam3d_manager._cached_model_data)}")
            print(f"_cached_model_data内容: {str(sam3d_manager._cached_model_data)[:500]}")
            
            cached_data = sam3d_manager._cached_model_data
            
            # 调试：如果是字符串，可能是glb格式
            if isinstance(cached_data, str):
                print("检测到glb路径，直接返回")
                return send_file(
                    cached_data,
                    mimetype='model/gltf-binary',
                    as_attachment=False,
                    download_name='model.glb'
                )
            
            import trimesh
            import io
            
            # 物体模型旋转矩阵：仅 Y-Z 轴交换，保持 X 轴方向不变以维持左右位置关系
            rotation_matrix = np.array([
                [1, 0, 0],
                [0, 0, 1],
                [0, -1, 0]
            ])
            
            all_meshes = []
                        
            # 确保 cached_data 是列表
            if not isinstance(cached_data, list):
                cached_data = [cached_data]
                        
            # 获取原始图片信息用于空间对齐
            image_path = getattr(sam3d_manager, '_last_image_path', None)
            img_width, img_height = 1024, 1024  # 默认值
            if image_path and os.path.exists(image_path):
                import cv2
                temp_img = cv2.imread(str(image_path))
                if temp_img is not None:
                    img_height, img_width = temp_img.shape[:2]
                        
            img_cx = img_width / 2.0
            img_cy = img_height / 2.0
                        
            # 获取缓存的物体检测数据（包含 bbox 信息）
            object_bboxes = []
            if hasattr(sam3d_manager, '_cached_multi_object_data') and sam3d_manager._cached_multi_object_data:
                for obj_data in sam3d_manager._cached_multi_object_data:
                    object_bboxes.append(obj_data.get('bbox', [0, 0, img_width, img_height]))
                        
            for obj_idx, result in enumerate(cached_data):
                print(f"处理物体 {obj_idx}, result type: {type(result)}")
                try:
                    if not isinstance(result, dict):
                        print(f"跳过非字典类型的结果")
                        continue
                        
                    mesh_list = result.get('mesh', [])
                    print(f"mesh_list type: {type(mesh_list)}, len: {len(mesh_list) if isinstance(mesh_list, list) else 'N/A'}")
                    
                    if isinstance(mesh_list, list) and len(mesh_list) > 0:
                        mesh = mesh_list[0]
                        print(f"mesh type: {type(mesh)}")
                    
                    if isinstance(mesh_list, list) and len(mesh_list) > 0:
                        mesh = mesh_list[0]
                        
                        if hasattr(mesh, 'vertices'):
                            vertices_np = mesh.vertices
                            if torch.is_tensor(vertices_np):
                                vertices_np = vertices_np.cpu().numpy()
                            vertices_rotated = vertices_np @ rotation_matrix.T
                            
                            if hasattr(mesh, 'faces'):
                                faces_np = mesh.faces
                                if torch.is_tensor(faces_np):
                                    faces_np = faces_np.cpu().numpy()
                            else:
                                faces_np = np.array([])
                            
                            # 应用空间位置补偿（使用 bbox 信息计算 2D 偏移）
                            vertices_offset = vertices_rotated.copy()
                            
                            # 如果有 bbox 信息，计算基于原图中心的偏移
                            if obj_idx < len(object_bboxes):
                                bbox = object_bboxes[obj_idx]
                                bbox_cx = (bbox[0] + bbox[2]) / 2.0
                                bbox_cy = (bbox[1] + bbox[3]) / 2.0
                                
                                # 假设一个固定的深度和焦距（物体没有 pred_cam_t）
                                # 使用类似多人体的算法，但调整系数以适应物体
                                focal_length = 1000.0  # 假设焦距
                                tz = 5.0  # 假设深度
                                dist_scale = 0.6  # 物体间距调节系数（调整到 0.6 以获得更自然的间距）
                                
                                tx_offset = (bbox_cx - img_cx) * tz / focal_length * dist_scale
                                # ty_offset = (bbox_cy - img_cy) * tz / focal_length * dist_scale  # Y轴不偏移，由地面对齐处理
                                
                                vertices_offset[:, 0] += tx_offset
                                print(f"物体 {obj_idx} 位置补偿: bbox=[{bbox[0]}, {bbox[2]}], bbox_cx={bbox_cx:.1f}, img_cx={img_cx:.1f}, tx_offset={tx_offset:.3f}")
                            
                            y_min = vertices_offset[:, 1].min()
                            vertices_offset[:, 1] -= y_min
                            
                            obj_mesh = trimesh.Trimesh(vertices=vertices_offset, faces=faces_np)
                            
                            if hasattr(mesh, 'visual'):
                                try:
                                    obj_mesh.visual = mesh.visual
                                except Exception as vis_err:
                                    print(f"保留材质失败 (物体 {obj_idx}): {vis_err}")
                            
                            all_meshes.append(obj_mesh)
                            print(f"物体 {obj_idx}: {len(vertices_np)} 顶点, {len(faces_np)} 面")
                        else:
                            print(f"物体 {obj_idx}: mesh没有vertices属性")
                    else:
                        vertices = result.get('pred_vertices')
                        vertex_colors = result.get('vertex_colors')
                        
                        if vertices is not None:
                            vertices_np = vertices.cpu().numpy() if torch.is_tensor(vertices) else vertices
                            vertices_rotated = vertices_np @ rotation_matrix.T
                            
                            y_min = vertices_rotated[:, 1].min()
                            vertices_offset = vertices_rotated.copy()
                            vertices_offset[:, 1] -= y_min
                            
                            try:
                                obj_mesh = trimesh.convex.convex_hull(vertices_offset)
                                
                                if vertex_colors is not None:
                                    colors_np = vertex_colors.cpu().numpy() if torch.is_tensor(vertex_colors) else vertex_colors
                                    if len(colors_np.shape) == 3 and colors_np.shape[2] == 3:
                                        colors_np = colors_np.reshape(-1, 3)
                                    colors_uint8 = (colors_np * 255).astype(np.uint8)
                                    
                                    obj_mesh.visual = trimesh.visual.ColorVisuals(
                                        obj_mesh,
                                        face_colors=np.full((len(obj_mesh.faces), 4), [128, 128, 128, 255], dtype=np.uint8)
                                    )
                                    
                                print(f"物体 {obj_idx}: 使用凸包创建网格，{len(obj_mesh.vertices)} 顶点, {len(obj_mesh.faces)} 面")
                            except Exception as hull_err:
                                print(f"凸包创建失败，使用点云: {hull_err}")
                                obj_mesh = trimesh.PointCloud(vertices=vertices_offset)
                                print(f"物体 {obj_idx}: 使用点云数据，{len(vertices_np)} 顶点")
                            
                            all_meshes.append(obj_mesh)
                except Exception as obj_err:
                    print(f"处理物体 {obj_idx} 时出错: {obj_err}")
                    continue
            
            if len(all_meshes) == 0:
                raise Exception('No valid mesh data found')
            
            if len(all_meshes) == 1:
                export_mesh = all_meshes[0]
            else:
                export_mesh = trimesh.util.concatenate(all_meshes)
                print(f"合并后: {len(export_mesh.vertices)} 顶点, {len(export_mesh.faces)} 面")
            
            buffer = io.BytesIO()
            export_mesh.export(buffer, file_type='glb')
            buffer.seek(0)
            
            return send_file(
                buffer,
                mimetype='model/gltf-binary',
                as_attachment=False,
                download_name='model.glb'
            )
        elif isinstance(sam3d_manager._cached_model_data, list) and len(sam3d_manager._cached_model_data) > 1:
            print(f"=== 返回多人3D模型 ===")
            print(f"人体数量: {len(sam3d_manager._cached_model_data)}")
            
            import trimesh
            import io
            
            meshes = []
            for idx, output_data in enumerate(sam3d_manager._cached_model_data):
                vertices = output_data.get("pred_vertices")
                faces = sam3d_manager._cached_model_faces
                vertex_colors = output_data.get("vertex_colors")
                pred_cam_t = output_data.get("pred_cam_t")

                vertices_np = vertices.cpu().numpy() if torch.is_tensor(vertices) else vertices
                faces_np = faces.cpu().numpy() if torch.is_tensor(faces) else faces

                # 人体模型旋转矩阵 (180° X轴)
                rotation_matrix = np.array([
                    [1, 0, 0],
                    [0, -1, 0],
                    [0, 0, -1]
                ])
                
                # 1. 旋转原始顶点
                vertices_rotated = vertices_np @ rotation_matrix.T
                
                # 2. 如果有相机平移，应用旋转后的平移以保持相对位置
                if pred_cam_t is not None:
                    t_np = pred_cam_t.cpu().numpy() if torch.is_tensor(pred_cam_t) else pred_cam_t
                    t_np = t_np.flatten()
                    if t_np.shape[0] == 3:
                        t_rotated = t_np @ rotation_matrix.T
                        vertices_rotated += t_rotated
                
                mesh = trimesh.Trimesh(vertices=vertices_rotated, faces=faces_np)
                
                if vertex_colors is not None:
                    vertex_colors_np = vertex_colors.cpu().numpy() if torch.is_tensor(vertex_colors) else vertex_colors
                    if len(vertex_colors_np.shape) == 2 and vertex_colors_np.shape[1] >= 3:
                        mesh.visual.vertex_colors = vertex_colors_np
                else:
                    mesh.visual = trimesh.visual.ColorVisuals(
                        mesh,
                        face_colors=np.full((len(faces_np), 4), [136, 136, 136, 255], dtype=np.uint8)
                    )
                
                meshes.append(mesh)
            
            combined_mesh = trimesh.util.concatenate(meshes)
            
            # 整体对齐到地面一次
            all_vertices = combined_mesh.vertices
            y_min = all_vertices[:, 1].min()
            combined_mesh.vertices[:, 1] -= y_min
            
            buffer = io.BytesIO()
            combined_mesh.export(buffer, file_type='glb')
            buffer.seek(0)
            
            return send_file(
                buffer,
                mimetype='model/gltf-binary',
                as_attachment=False,
                download_name='model.glb'
            )
        else:
            if model_type == 'objects':
                cached_data = sam3d_manager._cached_model_data
                if isinstance(cached_data, list) and len(cached_data) > 0:
                    output_data = cached_data[0]
                    vertices = output_data.get('pred_vertices')
                    vertex_colors = output_data.get('vertex_colors')
                elif isinstance(cached_data, dict):
                    output_data = cached_data
                    vertices = output_data.get('pred_vertices')
                    vertex_colors = output_data.get('vertex_colors')
                else:
                    vertices = None
                    vertex_colors = None
                faces = sam3d_manager._cached_model_faces
            else:
                output_data = sam3d_manager._cached_model_data
                vertices = output_data.get("pred_vertices")
                faces = sam3d_manager._cached_model_faces
                vertex_colors = output_data.get("vertex_colors")

            if vertices is None:
                return jsonify({'success': False, 'error': 'No vertex data available'})

            vertices_np = vertices.cpu().numpy() if torch.is_tensor(vertices) else vertices
            
            if faces is not None:
                faces_np = faces.cpu().numpy() if torch.is_tensor(faces) else faces
            else:
                faces_np = np.array([])

            if model_type == 'objects':
                rotation_matrix = np.array([
                    [1, 0, 0],
                    [0, 0, 1],
                    [0, -1, 0]
                ])
            else:
                rotation_matrix = np.array([
                    [1, 0, 0],
                    [0, -1, 0],
                    [0, 0, -1]
                ])

            vertices_rotated = vertices_np @ rotation_matrix.T

            y_min = vertices_rotated[:, 1].min()
            vertices_rotated = vertices_rotated.copy()
            vertices_rotated[:, 1] -= y_min

            import trimesh
            import io

            mesh = trimesh.Trimesh(vertices=vertices_rotated, faces=faces_np)
            
            if faces is None or len(faces_np) == 0:
                mesh = trimesh.PointCloud(vertices=vertices_rotated)

            if vertex_colors is not None:
                vertex_colors_np = vertex_colors.cpu().numpy() if torch.is_tensor(vertex_colors) else vertex_colors
                if len(vertex_colors_np.shape) == 2 and vertex_colors_np.shape[1] >= 3:
                    mesh.visual.vertex_colors = vertex_colors_np
                else:
                    mesh.visual = trimesh.visual.ColorVisuals(
                        mesh,
                        face_colors=np.full((len(faces_np), 4), [136, 136, 136, 255], dtype=np.uint8)
                    )
            else:
                mesh.visual = trimesh.visual.ColorVisuals(
                    mesh,
                    face_colors=np.full((len(faces_np), 4), [136, 136, 136, 255], dtype=np.uint8)
                )

            buffer = io.BytesIO()
            mesh.export(buffer, file_type='glb')
            buffer.seek(0)

            return send_file(
                buffer,
                mimetype='model/gltf-binary',
                as_attachment=False,
                download_name='model.glb'
            )
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/3d/skeleton', methods=['GET'])
def get_skeleton():
    """获取骨骼数据"""
    if sam3d_manager._cached_model_data is None:
        return jsonify({'success': False, 'error': 'No model data available'})

    try:
        # 检查是否是多人数据
        if isinstance(sam3d_manager._cached_model_data, list) and len(sam3d_manager._cached_model_data) > 1:
            # 多人模式，返回所有人的骨骼数据
            all_skeletons = []
            for idx, output_data in enumerate(sam3d_manager._cached_model_data):
                # 为每个人体添加偏移，与模型导出时的偏移一致
                offset_x = idx * 2.0  # 每个人体间隔2米
                skeleton = _extract_skeleton_data(output_data, idx, offset_x)
                if skeleton:
                    all_skeletons.append(skeleton)
            
            return jsonify({
                'success': True,
                'skeletons': all_skeletons,
                'person_count': len(all_skeletons)
            })
        else:
            # 单人模式
            output_data = sam3d_manager._cached_model_data
            skeleton = _extract_skeleton_data(output_data, 0, 0.0)
            
            if skeleton:
                return jsonify({
                    'success': True,
                    'skeletons': [skeleton],
                    'person_count': 1
                })
            else:
                return jsonify({'success': False, 'error': 'No skeleton data available'})
    except Exception as e:
        print(f"Error getting skeleton data: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

def _extract_skeleton_data(output_data, person_idx, offset_x=0.0):
    """提取单个人的骨骼数据"""
    # 获取关节位置
    joint_positions = output_data.get("pred_joint_coords")
    if joint_positions is None:
        joint_positions = output_data.get("pred_joints")

    if joint_positions is None:
        return None

    # 转换为numpy数组
    joint_positions_np = joint_positions.cpu().numpy() if torch.is_tensor(joint_positions) else joint_positions

    # 应用旋转（与模型旋转保持一致）
    model_type = sam3d_manager._cached_model_type or 'body'

    if model_type == 'objects':
        # 物体模型：需要-90° X轴旋转
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, 0, 1],
            [0, -1, 0]
        ])
    else:
        # 人体模型：应用180° X轴旋转
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]
        ])

    joint_positions_rotated = joint_positions_np @ rotation_matrix.T

    # 对齐到地面
    y_min = joint_positions_rotated[:, 1].min()
    joint_positions_aligned = joint_positions_rotated.copy()
    joint_positions_aligned[:, 1] -= y_min
    
    # 添加X轴偏移（与模型导出时的偏移一致）
    if offset_x != 0.0:
        joint_positions_aligned[:, 0] += offset_x
        print(f"人体 {person_idx} 骨骼添加偏移: {offset_x}")
    
    # 应用100倍缩放（与FBX导出时的缩放一致）
    # 注意：GLB导出时没有缩放，但骨骼需要与模型对齐
    # 前端加载的是GLB格式，所以不需要缩放
    # 如果前端加载的是FBX格式，则需要缩放
    # 这里保持原始缩放，因为前端显示的是GLB模型
    
    return {
        'person_index': person_idx,
        'joint_positions': joint_positions_aligned.tolist(),
        'num_joints': len(joint_positions_aligned)
    }

@app.route('/api/file/<path:filename>', methods=['GET'])
def serve_file(filename):
    try:
        return send_from_directory(OUTPUT_DIR, filename)
    except:
        return send_from_directory(COMFYUI_OUTPUT_DIR, filename)

@app.route('/api/comfyui/image/<path:filename>', methods=['GET'])
def serve_comfyui_image(filename):
    # 尝试多个可能的目录
    possible_dirs = [
        COMFYUI_OUTPUT_DIR,
        BASE_DIR.parent / "ComfyUI_windows_portable" / "ComfyUI" / "output",
        Path("D:/ComfyUI_windows_portable/ComfyUI/output"),
    ]

    for directory in possible_dirs:
        if directory.exists():
            filepath = directory / filename
            if filepath.exists():
                print(f"从目录提供图片: {directory}")
                return send_from_directory(str(directory), filename)

    print(f"⚠️ 图片文件不存在: {filename}")
    print(f"尝试的目录:")
    for directory in possible_dirs:
        print(f"  - {directory} (存在: {directory.exists()}, 包含文件: {(directory / filename).exists()})")

    return jsonify({'error': 'Image not found'}), 404

@app.route('/api/health', methods=['GET'])
def health_check():
    current_time = time.time()
    
    def get_time_info(last_used):
        if last_used is None:
            return {'loaded': False, 'last_used': None, 'idle_seconds': None}
        idle = int(current_time - last_used)
        return {'loaded': True, 'last_used': last_used, 'idle_seconds': idle}
    
    return jsonify({
        'status': 'ok',
        'torch_available': torch.cuda.is_available(),
        'sam3d_body': get_time_info(sam3d_manager._last_body_used),
        'sam3d_objects': get_time_info(sam3d_manager._last_objects_used)
    })

@app.route('/', methods=['GET'])
def index():
    response = send_from_directory(BASE_DIR, 'index_v2.html')
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/assets/<path:filename>', methods=['GET'])
def serve_assets(filename):
    return send_from_directory(BASE_DIR / 'assets', filename)

if __name__ == '__main__':
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)
