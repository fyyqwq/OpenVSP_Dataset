import openvsp as vsp
import os
import json
import numpy as np
import re
import subprocess
import time
from pathlib import Path
import random
import shutil
from typing import List, Dict, Any, Optional, Tuple
import glob
import sys
sys.path.append("./adb_parser/build/Release")

# 尝试导入adbparser，如果失败则设置为None
try:
    import adbparser
    ADBPARSER_AVAILABLE = True
except ImportError:
    print("⚠️ adbparser模块未找到，气动分析结果处理功能将被禁用")
    adbparser = None
    ADBPARSER_AVAILABLE = False

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import h5py
import psutil
import threading
import signal
from datetime import datetime
import logging
from enum import Enum


class ErrorType(Enum):
    """错误类型枚举"""
    GEOMETRY_ERROR = "geometry_error"
    MEMORY_ERROR = "memory_error"
    TIMEOUT_ERROR = "timeout_error"
    OPENVSP_ERROR = "openvsp_error"
    FILESYSTEM_ERROR = "filesystem_error"
    UNKNOWN_ERROR = "unknown_error"


class SamplingStatusTracker:
    """采样状态跟踪器"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.success_count = 0
        self.failure_count = 0
        self.target_count = 2000
        self.sampling_log = []
        self.error_stats = {error_type: 0 for error_type in ErrorType}
        
        # 创建日志文件
        self.log_file = os.path.join(output_dir, "sampling_log.json")
        self.status_file = os.path.join(output_dir, "sampling_status.json")
        
        # 加载已有状态（如果存在）
        self._load_status()
    
    def _load_status(self):
        """加载已有的采样状态"""
        if os.path.exists(self.status_file):
            try:
                with open(self.status_file, 'r') as f:
                    data = json.load(f)
                    self.success_count = data.get('success_count', 0)
                    self.failure_count = data.get('failure_count', 0)
                    self.sampling_log = data.get('sampling_log', [])
                    
                    # 修复error_stats的键类型问题
                    loaded_error_stats = data.get('error_stats', {})
                    self.error_stats = {error_type: 0 for error_type in ErrorType}
                    
                    # 将字符串键转换为枚举键
                    for error_type in ErrorType:
                        if error_type.value in loaded_error_stats:
                            self.error_stats[error_type] = loaded_error_stats[error_type.value]
                    
                print(f"📊 加载已有状态: 成功 {self.success_count}, 失败 {self.failure_count}")
            except Exception as e:
                print(f"⚠️ 加载状态文件失败: {e}")
                # 如果加载失败，重新初始化error_stats
                self.error_stats = {error_type: 0 for error_type in ErrorType}
    
    def record_success(self, sample_id: str, duration: float):
        """记录成功采样"""
        self.success_count += 1
        entry = {
            'sample_id': sample_id,
            'status': 'success',
            'duration': duration,
            'timestamp': datetime.now().isoformat(),
            'error_type': None,
            'error_message': None
        }
        self.sampling_log.append(entry)
        self._save_status()
        print(f"✅ 采样成功 {sample_id} (耗时: {duration:.2f}s) - 总计: {self.success_count}/{self.target_count}")
    
    def record_failure(self, sample_id: str, error_type: ErrorType, error_message: str, duration: float):
        """记录失败采样"""
        self.failure_count += 1
        self.error_stats[error_type] += 1
        
        entry = {
            'sample_id': sample_id,
            'status': 'failure',
            'duration': duration,
            'timestamp': datetime.now().isoformat(),
            'error_type': error_type.value,
            'error_message': error_message
        }
        self.sampling_log.append(entry)
        self._save_status()
        print(f"❌ 采样失败 {sample_id} ({error_type.value}: {error_message}) - 总计: {self.success_count}/{self.target_count}")
    
    def _save_status(self):
        """保存采样状态"""
        try:
            status_data = {
                'success_count': self.success_count,
                'failure_count': self.failure_count,
                'target_count': self.target_count,
                'sampling_log': self.sampling_log,
                'error_stats': {error_type.value: count for error_type, count in self.error_stats.items()},
                'last_updated': datetime.now().isoformat()
            }
            with open(self.status_file, 'w', encoding='utf-8') as f:
                json.dump(status_data, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            print(f"⚠️ 保存状态文件失败: {e}")
    
    def is_complete(self) -> bool:
        """检查是否达到目标数量"""
        return self.success_count >= self.target_count
    
    def get_progress(self) -> float:
        """获取进度百分比"""
        return (self.success_count / self.target_count) * 100 if self.target_count > 0 else 0
    
    def generate_report(self) -> Dict[str, Any]:
        """生成采样报告"""
        return {
            'total_sampled': self.success_count + self.failure_count,
            'success_count': self.success_count,
            'failure_count': self.failure_count,
            'success_rate': (self.success_count / (self.success_count + self.failure_count) * 100) if (self.success_count + self.failure_count) > 0 else 0,
            'error_distribution': {error_type.value: count for error_type, count in self.error_stats.items()},
            'progress': self.get_progress()
        }

class ResourceMonitor:
    """资源监控器"""
    
    def __init__(self, timeout_seconds: int = 60, memory_threshold: float = None):
        self.timeout_seconds = timeout_seconds
        self.process = psutil.Process()
        
        # 自动检测内存限制
        if memory_threshold is None:
            self.memory_threshold = self._auto_detect_memory_threshold()
        else:
            self.memory_threshold = memory_threshold
        
        print(f"💾 内存监控阈值设置为: {self.memory_threshold * 100:.1f}%")
    
    def _auto_detect_memory_threshold(self) -> float:
        """自动检测内存限制"""
        try:
            # 获取系统总内存
            total_memory = psutil.virtual_memory().total / (1024**3)  # GB
            
            # 根据系统内存大小设置阈值
            if total_memory >= 32:
                # 32GB+ 系统，可以设置较高阈值
                threshold = 0.85
            elif total_memory >= 16:
                # 16GB 系统
                threshold = 0.80
            elif total_memory >= 8:
                # 8GB 系统
                threshold = 0.75
            else:
                # 小于8GB 系统，设置较低阈值
                threshold = 0.70
            
            print(f"🖥️ 检测到系统内存: {total_memory:.1f}GB，设置阈值: {threshold * 100:.1f}%")
            return threshold
            
        except Exception as e:
            print(f"⚠️ 自动检测内存阈值失败: {e}，使用默认值 80%")
            return 0.80
    
    def check_memory(self) -> Tuple[bool, float]:
        """检查内存使用情况"""
        try:
            memory_percent = self.process.memory_percent()
            is_safe = memory_percent < (self.memory_threshold * 100)
            return is_safe, memory_percent
        except Exception:
            return True, 0.0
    
    def wait_for_file(self, file_path: str, timeout_seconds: int = None) -> bool:
        """等待文件生成，带超时"""
        if timeout_seconds is None:
            timeout_seconds = self.timeout_seconds
        
        start_time = time.time()
        while time.time() - start_time < timeout_seconds:
            if os.path.exists(file_path):
                return True
            time.sleep(0.5)
        return False
    
    def cleanup_vsp_state(self):
        """清理OpenVSP状态"""
        try:
            vsp.ClearVSPModel()
            vsp.Update()
        except Exception as e:
            print(f"⚠️ 清理VSP状态失败: {e}")
    
    def get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        try:
            memory = psutil.virtual_memory()
            cpu_count = psutil.cpu_count()
            
            return {
                'total_memory_gb': memory.total / (1024**3),
                'available_memory_gb': memory.available / (1024**3),
                'memory_percent': memory.percent,
                'cpu_count': cpu_count,
                'memory_threshold': self.memory_threshold
            }
        except Exception as e:
            print(f"⚠️ 获取系统信息失败: {e}")
            return {}

class ErrorClassifier:
    """错误分类器"""
    
    @staticmethod
    def classify_error(error: Exception, context: str = "") -> Tuple[ErrorType, str]:
        """分类错误类型"""
        error_str = str(error).lower()
        
        # 内存相关错误
        if any(keyword in error_str for keyword in ['memory', 'out of memory', 'malloc', 'segmentation fault']):
            return ErrorType.MEMORY_ERROR, f"内存错误: {error}"
        
        # OpenVSP相关错误
        if any(keyword in error_str for keyword in ['openvsp', 'vsp', 'segment fault', 'segfault']):
            return ErrorType.OPENVSP_ERROR, f"OpenVSP错误: {error}"
        
        # 几何相关错误
        if any(keyword in error_str for keyword in ['geometry', 'mesh', 'surface', 'invalid geometry']):
            return ErrorType.GEOMETRY_ERROR, f"几何错误: {error}"
        
        # 文件系统错误
        if any(keyword in error_str for keyword in ['file', 'permission', 'io', 'path']):
            return ErrorType.FILESYSTEM_ERROR, f"文件系统错误: {error}"
        
        # 超时错误
        if any(keyword in error_str for keyword in ['timeout', 'time out']):
            return ErrorType.TIMEOUT_ERROR, f"超时错误: {error}"
        
        return ErrorType.UNKNOWN_ERROR, f"未知错误: {error}"

class VSPBaseModel:
    """Base class for handling VSP model operations"""
    
    def __init__(self, vsp_file_path: str, output_dir: str = "vsp_dataset"):
        """
        Initialize the VSP model processor
        
        Args:
            vsp_file_path: Path to the .vsp3 file
            output_dir: Directory to store all generated data
        """
        self.vsp_file_path = vsp_file_path
        self.model_name = os.path.basename(vsp_file_path).split('.')[0]
        self.output_dir = output_dir
        self.model_dir = os.path.join(output_dir, self.model_name)
        
        # Create output directories
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(os.path.join(self.model_dir, "json"), exist_ok=True)
        os.makedirs(os.path.join(self.model_dir, "samples"), exist_ok=True)
        os.makedirs(os.path.join(self.model_dir, "mesh"), exist_ok=True)
        os.makedirs(os.path.join(self.model_dir, "pointcloud"), exist_ok=True)
        os.makedirs(os.path.join(self.model_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(self.model_dir, "aero"), exist_ok=True)
        
        # # Parameter list to extract (based on test12.py)
        # self.parameter_list = [
        #     # Global view parameters
        #     # "Zoom", "RotationX", "RotationY", "RotationZ", "ShowAxis", "ShowBorder",
        #     # Common geometry parameters
        #     "Tess_U", "Tess_W", "DrawType",
        #     # Pod-specific
        #     "Length", "FineRatio", "Diameter",
        #     # Wing-specific
        #     "Span", "Chord", "Sweep", "Twist", "Root_Chord", "Tip_Chord",
        #     # Fuselage-specific
        #     "RoundedRect_Width", "RoundedRect_Height", "Ellipse_Height", "Ellipse_Width",
        #     # Positioning and transformation
        #     "X_Location", "Y_Location", "Z_Location", "X_Rotation", "Y_Rotation", "Z_Rotation"
        #     # "Scale"
        # ]
        
        # Store model parameters
        self.model_params = {}
        
    def load_model(self):
        """Load the VSP model"""
        vsp.ClearVSPModel()
        vsp.ReadVSPFile(self.vsp_file_path)
        vsp.Update()
        
    def extract_parameters(self):
        """Extract parameters from the model and save to JSON"""
        pass
        
    def sample_models(self, num_samples):
        """Generate sample models by disturbing parameters"""
        pass
        
    def run_vspaero(self, model_path: str):
        """Run VSPAero analysis on a model"""
        pass
        
    def export_data(self, num_samples, index, model_path: str):
        """Export mesh, point cloud, and images"""
        pass
        
    def process(self):
        """Process the base model and generate the dataset"""
        pass


class VSPParameterExtractor(VSPBaseModel):
    """Class for extracting parameters from VSP models"""
    # Interested parameters
    PARAMETER_LIST = [
        "Zoom", "RotationX", "RotationY", "RotationZ", "ShowAxis", "ShowBorder",
        "Tess_U", "Tess_W", "DrawType",
        "Length", "FineRatio", "Diameter",
        "Span", "Chord", "Sweep", "Twist", "Root_Chord", "Tip_Chord", "Dihedral", "Area",
        "Ellipse_Width", "Ellipse_Height", "RoundedRect_Width", "RoundedRect_Height",
        "X_Rel_Location", "Y_Rel_Location", "Z_Rel_Location", 
        "X_Rel_Rotation", "Y_Rel_Rotation", "Z_Rel_Rotation", 
        "XLocPercent", "YLocPercent", "ZLocPercent",
        "XRotate", "YRotate", "ZRotate",
        "Scale", "TotalSpan", "TotalArea"
    ]
    GROUPS = [
        "Design", "Shape", "AdjustView", "XForm", "WingGeom", "Sym"
    ]

    def extract_parameters(self):
        """Extract parameters from the model and save to JSON"""
        # Load the model
        self.load_model()
        
        geom_ids = vsp.FindGeoms()
        geometry_list = []
        for geom_id in geom_ids:
            all_params = []
            all_pids = vsp.GetGeomParmIDs(geom_id)
            geom_name = vsp.GetGeomName(geom_id)
            geom_type = vsp.GetGeomTypeName(geom_id)
            for pid in all_pids:
                try:
                    group = vsp.GetParmGroupName(pid)
                    if group not in self.GROUPS:
                        continue
                    pname = vsp.GetParmName(pid)
                    if pname not in self.PARAMETER_LIST:
                        continue
                    val = vsp.GetParmVal(pid)
                    param_dict = {
                        "GroupName": group,
                        "Name": pname,
                        "Default": val
                    }
                    try:
                        lower = vsp.GetParmLowerLimit(pid)
                        upper = vsp.GetParmUpperLimit(pid)
                        if not (lower == 0.0 and upper == 0.0):
                            param_dict["Range"] = [lower, upper]
                    except:
                        pass
                    all_params.append(param_dict)
                except:
                    continue

            # 截面参数（适用于 Wing, Fuselage 等）
            try:
                xsec_surf_id = vsp.GetXSecSurf(geom_id, 0)
                num_sec = vsp.GetNumXSec(xsec_surf_id)
                geom_type = vsp.GetGeomTypeName(geom_id)
                start_idx = 1 if geom_type.upper() == "WING" else 0
                for i in range(start_idx, num_sec):
                    xsec_id = vsp.GetXSec(xsec_surf_id, i)
                    parm_ids = vsp.GetXSecParmIDs(xsec_id)
                    for pid in parm_ids:
                        try:
                            pname = vsp.GetParmName(pid)
                            if pname not in self.PARAMETER_LIST:
                                continue
                            group = vsp.GetParmGroupName(pid)
                            if group == "XSec":
                                group = f"XSec_{i}"
                            elif group == "XSecCurve":
                                group = f"XSecCurve_{i}"
                            else:
                                continue
                                
                            param_dict = {
                                "GroupName": group,
                                "Name": pname,
                                "Default": vsp.GetParmVal(pid)
                            }
                            try:
                                lower = vsp.GetParmLowerLimit(pid)
                                upper = vsp.GetParmUpperLimit(pid)
                                if not (lower == 0.0 and upper == 0.0):
                                    param_dict["Range"] = [lower, upper]
                            except:
                                pass
                            all_params.append(param_dict)
                        except:
                            continue
            except:
                pass
            
            geom_entry = {
                "Type": geom_type,
                "Params": all_params
            }
            geom_entry["Name"] = geom_name
            geometry_list.append(geom_entry)

        self.model_params = {
            "ModelName": self.model_name,
            "Geoms": geometry_list
        }
        return self.model_params
    
    
class VSPModelSampler(VSPBaseModel):
    """Class for sampling VSP models by disturbing parameters"""

    #Use "X_Rel_Location" to ensure proper positioning
    SAMPLE_PARAMETER_NAMES = [
        "Length", "X_Rel_Location", "Ellipse_Width", "Ellipse_Height",
        "Span", "Sweep", "Root_Chord", "Tip_Chord"
    ]

    SAMPLE_PARAMETER_POOLS = [
        "Span", "Sweep", "Twist", "Root_Chord", "Tip_Chord", "Dihedral",
        "Ellipse_Width", "Ellipse_Height",
        "X_Rel_Location", "Y_Rel_Location", "Z_Rel_Location", 
        "X_Rel_Rotation", "Y_Rel_Rotation", "Z_Rel_Rotation", 
        "XLocPercent", "YLocPercent", "ZLocPercent",
        "XRotate", "YRotate", "ZRotate",
        "Scale"
    ]

    SAMPLE_GROUPS = [
        "Design", "Shape", "XForm", "WingGeom", "XSec", "XSecCurve"
    ]

    def __init__(self, vsp_file_path: str, output_dir: str = "vsp_dataset", status_tracker: SamplingStatusTracker = None):
        super().__init__(vsp_file_path, output_dir)
        # 使用外部传入的状态跟踪器，如果没有则创建新的
        if status_tracker is not None:
            self.status_tracker = status_tracker
        else:
            self.status_tracker = SamplingStatusTracker(output_dir)
        self.resource_monitor = ResourceMonitor()
        self.output_dir = output_dir

    def sample_models(self, num_samples: int = 100, index: int = 0, target_success_count: int = 2000) -> List[str]:
        """Generate sample models by disturbing parameters with immediate validation"""
        # Make sure parameters are extracted
        if not self.model_params:
            self.extract_parameters()
            
        sample_paths = []
        sample_index = 0
        max_attempts = target_success_count * 10  # 设置最大尝试次数，避免无限循环
        
        print(f"🎯 开始逐个采样验证，目标成功数量: {target_success_count}")
        
        # 创建aero analyzer和data exporter
        aero_analyzer = VSPAeroAnalyzer(self.vsp_file_path, self.output_dir)
        data_exporter = VSPDataExporter(self.vsp_file_path, self.output_dir)
        
        while self.status_tracker.success_count < target_success_count and sample_index < max_attempts:
            sample_index += 1
            sample_name = f"{sample_index:06d}"
            sample_path = os.path.join(self.model_dir, "samples", f"{sample_name}.vsp3")
            
            start_time = time.time()
            
            try:
                # 检查是否已经达到目标数量
                if self.status_tracker.is_complete():
                    print(f"🎉 已达到目标成功数量: {self.status_tracker.success_count}")
                    break
                
                print(f"🔄 尝试采样 {sample_index}: {sample_name}")
                
                # 生成采样模型
                if sample_index == 1:
                    updated_params = self._apply_parameters(self.model_params)
                else:
                    # Create disturbed parameters
                    disturbed_params = self._disturb_parameters(self.model_params)
                    if disturbed_params is None:
                        raise Exception("参数扰动失败")
                        
                    # Apply disturbed parameters to model
                    updated_params = self._apply_parameters(disturbed_params)
                
                # 保存参数文件
                json_path1 = os.path.join(self.model_dir, "json", f"{sample_name}_params.json")
                with open(json_path1, "w") as f:
                    json.dump(updated_params, f, indent=2, sort_keys=False, default=str)
                
                json_path2 = os.path.join(self.model_dir, "json", f"{sample_name}_info.json")
                info_json = self._conclude_parameters(updated_params)
                with open(json_path2, "w") as f:
                    json.dump(info_json, f, indent=2, sort_keys=False, default=str)

                # Save the model
                vsp.WriteVSPFile(sample_path)
                
                # 验证模型文件是否生成
                if not os.path.exists(sample_path):
                    raise Exception("模型文件生成失败")
                
                print(f"  ✅ 模型生成成功，开始aero分析...")
                
                # 立即进行数据导出
                export_success = data_exporter.export_data(
                    num_samples=1, 
                    index=0, 
                    model_path=sample_path, 
                    timeout_seconds=150
                )
                if not export_success:
                    raise Exception("数据导出失败")
                
                
                # 立即进行aero分析
                aero_success = aero_analyzer.run_vspaero(sample_path, timeout_seconds=150)
                if not aero_success:
                    raise Exception("气动分析失败")
                
                print(f"  ✅ aero分析成功，开始数据导出...")
                               
                
                # 验证所有步骤都成功
                if self._verify_sample_completion(sample_name):
                    duration = time.time() - start_time
                    self.status_tracker.record_success(sample_name, duration)
                    sample_paths.append(sample_path)
                    print(f"  ✅ 采样 {sample_name} 完整验证成功 (耗时: {duration:.2f}s)")
                    print(f"  📊 当前进度: {self.status_tracker.success_count}/{target_success_count}")
                else:
                    raise Exception("采样验证失败")
                    
            except Exception as e:
                duration = time.time() - start_time
                error_type, error_msg = ErrorClassifier.classify_error(e, "采样验证")
                self.status_tracker.record_failure(sample_name, error_type, error_msg, duration)
                
                # 清理失败的采样文件
                self._cleanup_failed_sample(sample_name)
                
                print(f"  ❌ 采样 {sample_name} 验证失败: {error_msg}")
                print(f"  🔄 继续下一个采样...")
                
                # 清理VSP状态
                self.resource_monitor.cleanup_vsp_state()
                continue
        
        if sample_index >= max_attempts:
            print(f"⚠️ 达到最大尝试次数 {max_attempts}，停止采样")
        
        print(f"📊 采样完成: 成功 {self.status_tracker.success_count}, 失败 {self.status_tracker.failure_count}")
        return sample_paths
    
    def _verify_sample_completion(self, sample_name: str) -> bool:
        """验证采样是否完整完成"""
        try:
            # 检查必需的文件和目录
            required_paths = [
                os.path.join(self.model_dir, "json", f"{sample_name}_params.json"),
                os.path.join(self.model_dir, "json", f"{sample_name}_info.json"),
                os.path.join(self.model_dir, "samples", f"{sample_name}.vsp3"),
                os.path.join(self.model_dir, "pointcloud", f"{sample_name}_points.csv"),
            ]
            
            # 检查图像文件（至少一种格式）
            img_dir = os.path.join(self.model_dir, "images")           
            png_file = os.path.join(img_dir, f"{sample_name}.png")
            svg_file = os.path.join(img_dir, f"{sample_name}.svg")
            if not (os.path.exists(png_file) or os.path.exists(svg_file)):
                return False
            
            # 检查气动分析结果
            aero_dir = os.path.join(self.model_dir, "aero", sample_name)
            if os.path.exists(aero_dir):
                # 检查基本文件
                txt_files = glob.glob(os.path.join(aero_dir, "*.txt"))
                if not txt_files:
                    return False
                
                # 如果adbparser可用，检查H5文件
                if ADBPARSER_AVAILABLE:
                    h5_files = glob.glob(os.path.join(aero_dir, "*.h5"))
                    if not h5_files:
                        return False
            
            # 检查所有必需路径
            for path in required_paths:
                if not os.path.exists(path):
                    return False
            
            return True
            
        except Exception:
            return False
    
    def _cleanup_failed_sample(self, sample_name: str):
        """清理失败的采样文件"""
        try:
            # 清理各个目录中的失败文件
            directories = ["samples", "json", "mesh", "pointcloud", "images", "aero"]
            for directory in directories:
                dir_path = os.path.join(self.output_dir, directory)
                if os.path.exists(dir_path):
                    # 删除以sample_name开头的文件
                    for file in os.listdir(dir_path):
                        if file.startswith(sample_name):
                            file_path = os.path.join(dir_path, file)
                            if os.path.isfile(file_path):
                                os.remove(file_path)
                            elif os.path.isdir(file_path):
                                shutil.rmtree(file_path)
                                
        except Exception as e:
            print(f"⚠️ 清理失败采样文件时出错: {e}")
    
    def get_sampling_report(self) -> Dict[str, Any]:
        """获取采样报告"""
        return self.status_tracker.generate_report()

    def _conclude_parameters(self, params):
        info = {
            "AircraftName": self.model_name,
            "Configuration": "",
            "Engines": {
                "Number": 0,
                "Position": ""
            }
        }
        try:
            for geom in params["Geoms"]:
                geom_type = geom["Type"]
                geom_name = geom["Name"]

                if geom_type == "Wing":
                    info[geom_name] = []
                    geom_info = {
                        "Span": next((p['Default'] for p in geom['Params'] if p['Name'] == "TotalSpan"), 0.0),
                        "Area": next((p['Default'] for p in geom['Params'] if p['Name'] == "TotalArea"), 0.0)
                    }

                    geom_id = vsp.FindGeom(geom_name, 0)
                    xsec_surf_id = vsp.GetXSecSurf(geom_id, 0)
                    num_sec = vsp.GetNumXSec(xsec_surf_id)

                    root_chord, tip_chord, sweep = None, None, None
                    for p in geom['Params']:
                        group = p['GroupName']
                        name = p['Name']
                        if group == "XSec_1" and name == "Root_Chord":
                            root_chord = p['Default']
                        if group == f"XSec_{num_sec - 1}" and name == "Tip_Chord":
                            tip_chord = p['Default']
                        if group == "XSec_1" and name == "Sweep":
                            sweep = p['Default']

                    if root_chord and tip_chord:
                        geom_info["RootChord"] = root_chord
                        geom_info["TipChord"] = tip_chord
                    else:
                        geom_info["RootChord"] = 0.0
                        geom_info["TipChord"] = 0.0

                    geom_info["Sweep"] = sweep if sweep is not None else 0.0

                    info[geom_name].append(geom_info)

                elif geom_type == "Fuselage":
                    info[geom_name] = []
                    geom_info = {
                        "Length": next((p['Default'] for p in geom['Params'] if p['Name'] == "Length"), 0.0)
                    }

                    max_width = 0.0
                    max_height = 0.0

                    for p in geom['Params']:
                        group = p['GroupName']
                        name = p['Name']
                        if group.startswith("XSecCurve_"):
                            if name in ["Ellipse_Width", "RoundedRect_Width"]:
                                max_width = max(max_width, p['Default'])
                            if name in ["Ellipse_Height", "RoundedRect_Height"]:
                                max_height = max(max_height, p['Default'])

                    geom_info["MaxWidth"] = max_width
                    geom_info["MaxHeight"] = max_height
                    info[geom_name].append(geom_info)

                elif geom_type == "Pod":
                    info["Engines"]["Number"] += 1

        except Exception as e:
            print(f"⚠️ Error concluding parameters: {e}")
            return None

        return info

    def _disturb_parameters(self, original_params):
        """Create disturbed parameters based on original parameters"""
        
        try:
            # Deep copy of original parameters
            disturbed = json.loads(json.dumps(original_params))
            
            # Ensure 'geometries' key exists and is a list
            if "Geoms" not in disturbed or not isinstance(disturbed["Geoms"], list):
                raise KeyError("'Geoms' key missing or not a list")

            scale = random.uniform(-0.20, 0.20)
            width_ratio = random.uniform(-0.05, 0.05)
            height_ratio = random.uniform(-0.05, 0.05)
            main_span_ratio = random.uniform(-0.15, 0.15)
            hv_span_ratio = random.uniform(-0.15, 0.15)
            sweep_ratio = random.uniform(-0.15, 0.15)
            chord_ratio = random.uniform(-0.10, 0.10)

            # Disturb parameters for each geometry
            for geom in disturbed["Geoms"]:
                for param in geom['Params']:
                    group = param['GroupName']
                    name = param['Name']
                    value = param['Default']
                    if geom["Type"] == "Fuselage" and geom["Name"].lower() == "fuselage":
                        if name == "X_Rel_Location":
                            x_location = value
                        elif name == "Z_Rel_Location":
                            z_location = value
                    if geom["Name"].lower() == "horizontaltail" and name == "Z_Rel_Location":
                        param['Default'] = value * (1 + hv_span_ratio)
                            
            for geom in disturbed["Geoms"]:
                section_params = {}
                for param in geom['Params']:
                    group = param['GroupName']
                    name = param['Name']
                    value = param['Default']

                    if name not in self.SAMPLE_PARAMETER_NAMES or group.split('_')[0] not in self.SAMPLE_GROUPS:
                        continue

                    if name == "Length":
                        param['Default'] = value * (1 + scale)
                        continue
                    
                    # ensure proper positioning
                    if name == "X_Rel_Location":
                        if geom["Type"] == "Fuselage":
                            continue
                        param['Default'] = x_location + value * (1 + scale)
                        continue
                    
                    if name == "Sweep":
                        if value >= 60.0 and sweep_ratio not in [-0.02, 0.02]:
                            disturbance = max(-0.02, min(sweep_ratio, 0.02))
                            param['Default'] = value * (1 + disturbance)
                            continue
                        if value <= 2.0:
                            param['Default'] = random.uniform(0.0, 5.0)  
                            continue
                        param['Default'] = value * (1 + sweep_ratio)
                        continue

                    if name == "Span" and geom['Name'].lower() == "mainwing":
                        param["Default"] = value * (1 + main_span_ratio)
                        continue

                    if name == "Span" and geom['Name'].lower().startswith(("horizontaltail", "verticaltail")):
                        param["Default"] = value * (1 + hv_span_ratio)
                        continue

                    if name == "Ellipse_Width":
                        param["Default"] = value * (1 + width_ratio)
                        continue

                    if name == "Ellipse_Height":
                        param["Default"] = value * (1 + height_ratio)
                        continue

                    if geom["Type"] == "Fuselage" and name == "XLocPercent":
                        if param['Default'] == 1.0:
                            continue

                    if geom["Type"] == "Wing" and name in ["Root_Chord", "Tip_Chord"]:
                        sec_idx = int(group.split('_')[1])
                        section_params.setdefault(sec_idx, []).append(param)
                        if name == "Root_Chord" and sec_idx == 1:
                            param['Default'] = value * (1 + chord_ratio)
                        elif name == "Tip_Chord":
                            param['Default'] = value * (1 + chord_ratio)
                        continue

                    # Disturb parameter by random percentage
                    disturbance = random.uniform(-0.15, 0.15)
                    param['Default'] = value * (1 + disturbance)

                sorted_sections = sorted(section_params.keys())
                # 强制参数一致性（如：Tip_Chord[i] = Root_Chord[i+1]）
                for i in range(1, len(sorted_sections)):
                    tip_chord = next((p for p in section_params[i] if p["Name"] == "Tip_Chord"), None)
                    next_root_chord = next((p for p in section_params[i + 1] if p["Name"] == "Root_Chord"), None)
                    if tip_chord and next_root_chord:
                        next_root_chord["Default"] = tip_chord["Default"]

            return disturbed

        except Exception as e:
            print(f"⚠️ Error disturbing parameters: {e}")
            return None
    
    def _apply_parameters(self, params):
        """Apply parameters to the current model"""
        try:
            self.load_model()
            # First pass: per-section parameters only
            for geom in params["Geoms"]:
                geom_id = vsp.FindGeom(geom['Name'], 0)
                per_geom_overall = []  # Store total-level parameters temporarily

                for param in geom['Params']:
                    group = param['GroupName']
                    name = param['Name']
                    value = param['Default']

                    if name == "Z_Rel_Location" and geom['Name'].lower().startswith(("horizontaltail")):
                        parm_id = vsp.FindParm(geom_id, name, group)
                        if parm_id:
                            vsp.SetParmVal(parm_id, value)
                            continue

                    if name not in self.SAMPLE_PARAMETER_NAMES or group.split('_')[0] not in self.SAMPLE_GROUPS:
                        continue

                    if name == "Length" and group == "Design":
                        parm_id = vsp.FindParm(geom_id, name, group)
                        if parm_id:
                            vsp.SetParmVal(parm_id, value)
                            continue
                        else:
                            print(f"⚠️ Parameter {name} not found in group {group} for geometry {geom['Name']}")
                            continue
                    if group.startswith("XSec_") or group.startswith("XSecCurve_"):
                        # Section-level: apply immediately
                        sec_index = int(group.split('_')[1])
                        xsec_surf_id = vsp.GetXSecSurf(geom_id, 0)
                        xsec_id = vsp.GetXSec(xsec_surf_id, sec_index)
                        parm_id = vsp.GetXSecParm(xsec_id, name)
                        vsp.SetParmVal(parm_id, value)
                    else:
                        # Global-level: delay setting
                        per_geom_overall.append((group, name, value, geom_id))

                # Second pass: apply overall/global parameters
                for group, name, value, geom_id in per_geom_overall:
                    parm_id = vsp.FindParm(geom_id, name, group)
                    if parm_id:
                        vsp.SetParmVal(parm_id, value)
                vsp.Update()
                if geom["Type"] == "Wing":
                    total_span = vsp.GetParmVal(vsp.FindParm(geom_id, "TotalSpan", "WingGeom"))
                    total_area = vsp.GetParmVal(vsp.FindParm(geom_id, "TotalArea", "WingGeom"))
                    for param in geom['Params']:
                        if param['Name'] == "TotalSpan":
                            param['Default'] = total_span
                        elif param['Name'] == "TotalArea":
                            param['Default'] = total_area

                    xsec_surf_id = vsp.GetXSecSurf(geom_id, 0)
                    num_sec = vsp.GetNumXSec(xsec_surf_id)
                    for i in range(1, num_sec):
                        sect_area = vsp.GetParmVal(vsp.FindParm(geom_id, "Area", f"XSec_{i}"))
                        for param in geom['Params']:
                            if param['Name'] == "Area" and param['GroupName'] == f"XSec_{i}":
                                param['Default'] = sect_area

            # vsp.Update()

        except Exception as e:
            print(f"❌ Failed to apply parameters: {e}")

        return params
    
    
class VSPAeroAnalyzer(VSPBaseModel):
    """Class for running VSPAero analysis on VSP models"""
    
    def __init__(self, vsp_file_path: str, output_dir: str = "vsp_dataset"):
        super().__init__(vsp_file_path, output_dir)
        self.resource_monitor = ResourceMonitor()
    
    def run_vspaero(self, model_path: Optional[str] = None, timeout_seconds: int = 60) -> bool:
        """Run VSPAero analysis on a model with timeout and error handling"""
        start_time = time.time()
        
        if model_path:
            # Load the specified model
            try:
                vsp.ClearVSPModel()
                vsp.ReadVSPFile(model_path)
                vsp.Update()
                model_name = os.path.basename(model_path).split('.')[0]
            except Exception as e:
                error_type, error_msg = ErrorClassifier.classify_error(e, "加载模型")
                raise Exception(f"加载模型失败: {error_msg}")
        else:
            # Use the base model
            self.load_model()
            model_name = self.model_name
            
        try:
            # Check memory before starting
            is_safe, memory_percent = self.resource_monitor.check_memory()
            if not is_safe:
                raise Exception(f"内存使用率过高: {memory_percent:.1f}%")
            
            # Check if VSPAero is available in this OpenVSP version
            if not hasattr(vsp, 'SetAnalysisInputDefaults'):
                print("Warning: VSPAero analysis not available in this OpenVSP version")
                print("Skipping VSPAero analysis for this model")
                return True
                
            # Setup VSPAero analysis
            mesh_name = "VSPAEROComputeGeometry"
            analysis_name = "VSPAEROSweep"
            
            # Set mesh parameters with timeout
            def setup_mesh():
                vsp.SetAnalysisInputDefaults(mesh_name)
                vsp.SetIntAnalysisInput(mesh_name, "Set", [-3])
                vsp.SetAnalysisInputDefaults(analysis_name)
                analysis_method = vsp.GetIntAnalysisInput(analysis_name, "AnalysisMethod")
                analysis_method = [vsp.PANEL]
                vsp.SetIntAnalysisInput(mesh_name, "AnalysisMethod", analysis_method)
                vsp.SetIntAnalysisInput(analysis_name, "AnalysisMethod", analysis_method)
                vsp.Update()
            
            # Execute setup with timeout
            self._execute_with_timeout(setup_mesh, timeout_seconds // 2, "设置网格参数")

            # Set analysis parameters
            def setup_analysis():
                vsp.SetIntAnalysisInput(analysis_name, "Set", [-3])
                vsp.SetIntAnalysisInput(analysis_name, "RefFlag", [1])
                vsp.SetIntAnalysisInput(analysis_name, "Xcg", [1])
                vsp.SetIntAnalysisInput(analysis_name, "MachNpts", [3])
                vsp.SetDoubleAnalysisInput(analysis_name, "MachStart", [0.1])
                vsp.SetDoubleAnalysisInput(analysis_name, "MachEnd", [0.8])
                vsp.SetDoubleAnalysisInput(analysis_name, "ReCref", [1e7])
                vsp.SetIntAnalysisInput(analysis_name, "AlphaNpts", [3])
                vsp.SetDoubleAnalysisInput(analysis_name, "AlphaStart", [0.0])
                vsp.SetDoubleAnalysisInput(analysis_name, "AlphaEnd", [10.0])
                vsp.SetIntAnalysisInput(analysis_name, "BetaNpts", [1])
                vsp.SetDoubleAnalysisInput(analysis_name, "BetaStart", [0.0])
                vsp.SetDoubleAnalysisInput(analysis_name, "BetaEnd", [0.0])
                vsp.SetIntAnalysisInput(analysis_name, "WakeNumIter", [5])
                vsp.SetIntAnalysisInput(analysis_name, "NCPU", [8])
                vsp.SetIntAnalysisInput(analysis_name, "NumWakeNodes", [64])
                vsp.Update()
            
            self._execute_with_timeout(setup_analysis, timeout_seconds // 4, "设置分析参数")
            
            vspaero_dir = os.path.join(self.model_dir, "samples") if model_path else os.path.dirname(self.vsp_file_path)
            
            # Run mesh computation with timeout
            def run_mesh():
                vsp.ExecAnalysis(mesh_name)
            
            self._execute_with_timeout(run_mesh, timeout_seconds // 2, "网格计算")
            
            # Run analysis with timeout
            def run_analysis():
                vsp.ExecAnalysis(analysis_name)
            
            self._execute_with_timeout(run_analysis, timeout_seconds, "气动分析")

            # Move results and verify
            aero_dir = os.path.join(self.model_dir, "aero", model_name)
            os.makedirs(aero_dir, exist_ok=True)
            
            # Wait for result files with timeout
            expected_files = ['*.adb', '*.adb.cases', '*.polar']
            all_files_exist = True
            for pattern in expected_files:
                files = glob.glob(os.path.join(vspaero_dir, pattern))
                if not files:
                    all_files_exist = False
                    break
            
            if not all_files_exist:
                raise Exception("气动分析结果文件未生成")
            
            # Process results
            self._get_pressure_results(vspaero_dir, aero_dir)
            
            # Move files
            for file in os.listdir(vspaero_dir):
                full_path = os.path.join(vspaero_dir, file)
                if file.endswith(".vsp3"):
                    continue
                if not (file.endswith('.h5') or file.endswith('.obj') or file.endswith('.txt') or file.endswith('.polar')):
                    os.remove(full_path)
                    continue
                src = full_path
                dst = os.path.join(aero_dir, file)
                shutil.move(src, dst)
            
            # Verify final results
            if not self._verify_aero_results(aero_dir):
                raise Exception("气动分析结果验证失败")
                
            print(f"✅ VSPAero analysis completed for {model_name}")
            return True
                    
        except Exception as e:
            duration = time.time() - start_time
            error_type, error_msg = ErrorClassifier.classify_error(e, "气动分析")
            print(f"❌ VSPAero analysis failed for {model_name}: {error_msg}")
            self.resource_monitor.cleanup_vsp_state()
            raise Exception(f"气动分析失败: {error_msg}")
    
    def _execute_with_timeout(self, func, timeout_seconds: int, operation_name: str):
        """执行函数并设置超时"""
        result = [None]
        exception = [None]
        
        def target():
            try:
                result[0] = func()
            except Exception as e:
                exception[0] = e
        
        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        thread.join(timeout_seconds)
        
        if thread.is_alive():
            raise Exception(f"{operation_name}超时 ({timeout_seconds}s)")
        
        if exception[0]:
            raise exception[0]
        
        return result[0]
    
    def _verify_aero_results(self, aero_dir: str) -> bool:
        """验证气动分析结果"""
        try:
            # 检查基本文件是否存在
            required_files = ['*.txt']
            for pattern in required_files:
                files = glob.glob(os.path.join(aero_dir, pattern))
                if not files:
                    return False
            
            # 如果adbparser可用，检查H5文件内容
            if ADBPARSER_AVAILABLE:
                h5_files = glob.glob(os.path.join(aero_dir, "*.h5"))
                for h5_file in h5_files:
                    try:
                        with h5py.File(h5_file, 'r') as hf:
                            required_datasets = ['points', 'cps', 'mach', 'aoa', 'cl', 'cd']
                            for dataset in required_datasets:
                                if dataset not in hf:
                                    return False
                    except Exception:
                        return False
            else:
                # 如果没有adbparser，只检查基本文件
                print("⚠️ adbparser不可用，跳过H5文件验证")
            
            return True
        except Exception:
            return False

    def _get_pressure_results(self, results_dir, aero_dir):
        """处理气动分析结果，提取压力数据"""
        if not ADBPARSER_AVAILABLE:
            print("⚠️ adbparser不可用，跳过压力结果处理")
            return
        
        try:
            model_name = os.path.basename(aero_dir).split('.')[0]
            case_info_list = []
            adb_file = glob.glob(os.path.join(results_dir, "*.adb"))[0]
            case_file = glob.glob(os.path.join(results_dir, "*.adb.cases"))[0]
            polar_file = glob.glob(os.path.join(results_dir, "*.polar"))[0]
            vspgeom_file = glob.glob(os.path.join(results_dir, "*.vspgeom"))[0]
            adb_filename = os.path.basename(adb_file)
            base_name = os.path.splitext(adb_filename)[0]

            # vspgeom -> obj
            with open(vspgeom_file, "r") as f:
                lines = f.readlines()

            i = 0
            n_points = int(lines[i].strip())
            i += 1

            vertices = []
            for _ in range(n_points):
                parts = lines[i].strip().split()
                if len(parts) == 3:
                    vertices.append(tuple(map(float, parts)))
                i += 1

            n_faces = int(lines[i].strip())
            i += 1

            faces = []
            for _ in range(n_faces):
                parts = lines[i].strip().split()
                if parts[0] == "3":
                    face = tuple(int(idx) for idx in parts[1:4])
                    faces.append(face)
                i += 1

            obj_file = os.path.join(results_dir, f"{model_name}.obj")
            with open(obj_file, "w") as f:
                for v in vertices:
                    f.write(f"v {v[0]} {v[1]} {v[2]}\n")
                for face in faces:
                    f.write(f"f {face[0]} {face[1]} {face[2]}\n")

            slicer = adbparser.ADBSlicer()
            slicer.load_file(os.path.join(results_dir, base_name))
            try:
                with open(case_file, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 3:
                            mach = float(parts[0])
                            aoa = float(parts[1])
                            beta = float(parts[2])
                            case_info_list.append((mach, aoa, beta))
            except FileNotFoundError:
                print("Only one case found")
                pass

            try:
                for case_id, (mach, aoa, beta) in enumerate(case_info_list, start=1):
                    with open(polar_file, "r") as f:
                        line = f.readlines()[case_id]
                        if line.strip() == "":
                            continue
                        parts = line.split()
                        mach = float(parts[1])
                        aoa = float(parts[2])
                        cl = float(parts[4])
                        cd = float(parts[7])
                    slicer.load_solution(case_id)
                    num_tris = slicer.get_triangle_count()
                    output_file = os.path.join(aero_dir, f"{model_name}_case{case_id}.txt")
                    points = []
                    cps = []
                    with open(output_file, "w") as f:
                        f.write(f"mach: {mach:.6f}, aoa: {aoa:.6f}, cl: {cl:.6f}, cd: {cd:.6f}\n")
                        f.write("coords,cp\n")
                        for tri_id in range(1, num_tris + 1):
                            try:
                                cp = slicer.get_cp(tri_id)
                                n1, n2, n3, _ = slicer.get_triangle(tri_id)
                                x1, y1, z1 = slicer.get_node(n1)
                                x2, y2, z2 = slicer.get_node(n2)
                                x3, y3, z3 = slicer.get_node(n3)
                                x = (x1 + x2 + x3) / 3.0
                                y = (y1 + y2 + y3) / 3.0
                                z = (z1 + z2 + z3) / 3.0
                                coords = (x, y, z)
                                f.write(f"{coords},{cp:.6f},\n")
                                points.append(coords)
                                cps.append(cp)
                            except Exception as e:
                                print(f"结束于 Triangle {tri_id}，原因: {e}")
                                break

                        points = np.expand_dims(points, axis=0)
                        cps = np.expand_dims(cps, axis=0)
                        filename = f"{model_name}{case_id:02d}.h5"
                        h5_file = os.path.join(aero_dir, filename)
                        with h5py.File(h5_file, 'w') as hf:
                            hf.create_dataset('points', data=points.astype('float32'))
                            hf.create_dataset('cps', data=cps.astype('float32'))
                            hf.create_dataset('mach', data=np.array([mach], dtype='float32'))
                            hf.create_dataset('aoa', data=np.array([aoa], dtype='float32'))
                            hf.create_dataset('cl', data=np.array([cl], dtype='float32'))
                            hf.create_dataset('cd', data=np.array([cd], dtype='float32'))
            except Exception as e:
                print(f"Error processing ADB files: {str(e)}")
                return []
        except Exception as e:
            print(f"Error in pressure results processing: {str(e)}")
            return []


    def _extract_aero_results(self, results_dir):
        """Extract key results from VSPAero output"""
        results = {
            "cl": [],
            "cd": [],
            "surface_pressure": []
        }
        
        # Parse history file for CL/CD
        history_file = os.path.join(results_dir, "history.csv")
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                lines = f.readlines()
                # Skip header
                for line in lines[1:]:
                    values = line.strip().split(',')
                    if len(values) >= 3:
                        try:
                            alpha = float(values[0])
                            cl = float(values[1])
                            cd = float(values[2])
                            results["cl"].append((alpha, cl))
                            results["cd"].append((alpha, cd))
                        except:
                            pass
        return results


class VSPDataExporter(VSPBaseModel):
    """Class for exporting VSP model data (mesh, point cloud, images)"""
    
    def __init__(self, vsp_file_path: str, output_dir: str = "vsp_dataset"):
        super().__init__(vsp_file_path, output_dir)
        self.resource_monitor = ResourceMonitor()
    
    def export_data(self, num_samples: int = 100, index: int = 0, model_path: Optional[str] = None, timeout_seconds: int = 60) -> bool:
        """Export mesh, point cloud, and images with timeout and error handling"""
        start_time = time.time()
        
        if model_path:
            # Load the specified model
            try:
                vsp.ClearVSPModel()
                vsp.ReadVSPFile(model_path)
                vsp.Update()
                model_name = os.path.basename(model_path).split('.')[0]
            except Exception as e:
                error_type, error_msg = ErrorClassifier.classify_error(e, "加载模型")
                raise Exception(f"加载模型失败: {error_msg}")
        else:
            # Use the base model
            self.load_model()
            model_name = f"{1+index*(num_samples+1):06d}"

        try:
            # Check memory before starting
            is_safe, memory_percent = self.resource_monitor.check_memory()
            if not is_safe:
                raise Exception(f"内存使用率过高: {memory_percent:.1f}%")

            # Export mesh (disabled due to memory issues)
            # mesh_success = self._export_mesh_with_timeout(model_name, timeout_seconds)
            # if not mesh_success:
            #     raise Exception("网格导出失败")

            # Export point cloud
            # pc_success = self._export_point_cloud_with_timeout(model_name, timeout_seconds)
            # if not pc_success:
            #     raise Exception("点云导出失败")
            
            # Export multi-view images
            img_success = self._export_images_with_timeout(model_name, timeout_seconds)
            if not img_success:
                raise Exception("图像导出失败")
            
            # Verify all exports
            # if not self._verify_exports(model_name):
            #     raise Exception("导出结果验证失败")
            
            print(f"✅ Data export completed for {model_name}")
            return True
            
        except Exception as e:
            duration = time.time() - start_time
            error_type, error_msg = ErrorClassifier.classify_error(e, "数据导出")
            print(f"❌ Data export failed for {model_name}: {error_msg}")
            self.resource_monitor.cleanup_vsp_state()
            raise Exception(f"数据导出失败: {error_msg}")
    
    def _export_mesh_with_timeout(self, model_name: str, timeout_seconds: int) -> bool:
        """Export model mesh with timeout"""
        print("Exporting mesh...")
        mesh_dir = os.path.join(self.model_dir, "mesh")
        os.makedirs(mesh_dir, exist_ok=True)

        try:
            def export_mesh():
                # Check memory before mesh generation
                is_safe, memory_percent = self.resource_monitor.check_memory()
                if not is_safe:
                    raise Exception(f"内存使用率过高: {memory_percent:.1f}%")
                
                vsp.SetCFDMeshVal(vsp.CFD_MIN_EDGE_LEN, 0.10)
                vsp.SetCFDMeshVal(vsp.CFD_MAX_EDGE_LEN, 0.50)
                vsp.SetCFDMeshVal(vsp.CFD_MAX_GAP, 0.05)
                vsp.SetCFDMeshVal(vsp.CFD_NUM_CIRCLE_SEGS, 16.0)
                vsp.SetCFDMeshVal(vsp.CFD_GROWTH_RATIO, 1.3)
                vsp.SetCFDMeshVal(vsp.CFD_LIMIT_GROWTH_FLAG, 1)
                vsp.SetCFDMeshVal(vsp.CFD_INTERSECT_SUBSURFACE_FLAG, 1)
                vsp.SetCFDMeshVal(vsp.CFD_LIMIT_GROWTH_FLAG, 1)
                
                stl_file = os.path.join(mesh_dir, f"{model_name}.stl")
                vsp.SetComputationFileName(vsp.CFD_STL_TYPE, stl_file)
                vsp.ComputeCFDMesh(vsp.SET_ALL, vsp.SET_NONE, vsp.CFD_STL_TYPE)
                
                # Wait for STL file
                if not self.resource_monitor.wait_for_file(stl_file, timeout_seconds // 2):
                    raise Exception("STL文件生成超时")
                
                obj_file = os.path.join(mesh_dir, f"{model_name}.obj")
                vsp.SetComputationFileName(vsp.CFD_OBJ_TYPE, obj_file)
                vsp.ComputeCFDMesh(vsp.SET_ALL, vsp.SET_NONE, vsp.CFD_OBJ_TYPE)
                
                # Wait for OBJ file
                if not self.resource_monitor.wait_for_file(obj_file, timeout_seconds // 2):
                    raise Exception("OBJ文件生成超时")
                
                print(f"Mesh exported to {mesh_dir}")
                return True
            
            return self._execute_with_timeout(export_mesh, timeout_seconds, "网格导出")
            
        except Exception as e:
            print(f"Error exporting mesh: {str(e)}")
            return False
    
    def _export_point_cloud_with_timeout(self, model_name: str, timeout_seconds: int) -> bool:
        """Export model point cloud with timeout"""
        pc_dir = os.path.join(self.model_dir, "pointcloud")
        os.makedirs(pc_dir, exist_ok=True)
        
        try:
            def export_point_cloud():
                # Check memory before export
                is_safe, memory_percent = self.resource_monitor.check_memory()
                if not is_safe:
                    raise Exception(f"内存使用率过高: {memory_percent:.1f}%")
                
                # Export point cloud (CSV format)
                csv_file = os.path.join(pc_dir, f"{model_name}_points.csv")
                
                # Use mesh export and convert to point cloud
                dat_file = os.path.join(pc_dir, f"{model_name}.dat")
                vsp.ExportFile(dat_file, vsp.SET_ALL, vsp.EXPORT_NASCART)
                
                # Wait for DAT file
                if not self.resource_monitor.wait_for_file(dat_file, timeout_seconds // 2):
                    raise Exception("DAT文件生成超时")
                
                points = []
                with open(dat_file, 'r') as f:
                    next(f)
                    for line in f:
                        # 跳过空行
                        if not line.strip():
                            continue
                        # 尝试分割每行为3个浮点数
                        parts = line.split()
                        if len(parts) != 3:
                            # 当遇到非点云数据行时停止读取
                            break
                            
                        try:
                            # 尝试转换所有部分为浮点数
                            coords = [float(part) for part in parts]
                            points.append(coords)
                        except ValueError:
                            # 当遇到无法转换的行时停止读取
                            break
                
                with open(csv_file, 'w') as f:
                    f.write("x,y,z\n")
                    for point in points:
                        f.write(f"{point[0]},{point[1]},{point[2]}\n")
                
                # Clean up temporary files
                if os.path.exists(dat_file):
                    os.remove(dat_file)
                    os.remove(dat_file.replace('.dat', '.key'))
                    os.remove(dat_file.replace('.dat', '.tkey'))
                
                print(f"Point cloud exported to {pc_dir}")
                return True
            
            return self._execute_with_timeout(export_point_cloud, timeout_seconds, "点云导出")
            
        except Exception as e:
            print(f"Error exporting point cloud: {str(e)}")
            return False
    
    def _export_images_with_timeout(self, model_name: str, timeout_seconds: int) -> bool:
        """Export multi-view images with timeout"""
        img_dir = os.path.join(self.model_dir, "images")
        os.makedirs(img_dir, exist_ok=True)
        
        # Define standard views
        views = {
            "front": vsp.VIEW_FRONT,
            "top": vsp.VIEW_TOP,
            "side": vsp.VIEW_LEFT,
        }
        
        try:
            def export_images():
                # Check memory before export
                is_safe, memory_percent = self.resource_monitor.check_memory()
                if not is_safe:
                    raise Exception(f"内存使用率过高: {memory_percent:.1f}%")
                    
                # Export as SVG as fallback
                svg_file = os.path.join(img_dir, f"{model_name}.svg")
                vsp.ExportFile(svg_file, vsp.SET_ALL, vsp.EXPORT_SVG)
                
                # Wait for SVG file
                if not self.resource_monitor.wait_for_file(svg_file, timeout_seconds // 3):
                    raise Exception(f"{model_name} SVG生成超时")
                
                print(f"Exported {model_name} view as SVG: {svg_file}")
                
                print(f"Images exported to {img_dir}")
                return True
            
            return self._execute_with_timeout(export_images, timeout_seconds, "图像导出")
            
        except Exception as e:
            print(f"Error exporting images: {str(e)}")
            return False
    
    def _execute_with_timeout(self, func, timeout_seconds: int, operation_name: str):
        """执行函数并设置超时"""
        result = [None]
        exception = [None]
        
        def target():
            try:
                result[0] = func()
            except Exception as e:
                exception[0] = e
        
        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        thread.join(timeout_seconds)
        
        if thread.is_alive():
            raise Exception(f"{operation_name}超时 ({timeout_seconds}s)")
        
        if exception[0]:
            raise exception[0]
        
        return result[0]
    
    def _verify_exports(self, model_name: str) -> bool:
        """验证导出结果"""
        try:
            # 检查点云文件
            pc_file = os.path.join(self.model_dir, "pointcloud", f"{model_name}_points.csv")
            if not os.path.exists(pc_file):
                return False
            
            # 检查图像文件（PNG或SVG）
            img_dir = os.path.join(self.model_dir, "images")
            png_file = os.path.join(img_dir, f"{model_name}.png")
            svg_file = os.path.join(img_dir, f"{model_name}.svg")
            if not (os.path.exists(png_file) or os.path.exists(svg_file)):
                return False
            
            return True
        except Exception:
            return False


class VSPDatasetGenerator:
    """Main class for generating VSP datasets"""
    
    def __init__(self, vsp_files: List[str], output_dir: str = "vsp_dataset"):
        """
        Initialize the dataset generator
        
        Args:
            vsp_files: List of paths to .vsp3 files
            output_dir: Directory to store all generated data
        """
        self.vsp_files = vsp_files
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建全局状态跟踪器
        self.global_status_tracker = SamplingStatusTracker(output_dir)
        self.resource_monitor = ResourceMonitor()
        
        # 设置中断信号处理
        self._setup_interrupt_handler()
    
    def _setup_interrupt_handler(self):
        """设置中断信号处理"""
        def signal_handler(signum, frame):
            print(f"\n⚠️ 收到中断信号 {signum}，正在保存状态...")
            self._save_interrupt_state()
            print("💾 状态已保存，可以安全退出")
            sys.exit(0)
        
        try:
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
        except Exception as e:
            print(f"⚠️ 设置中断处理器失败: {e}")
    
    def _save_interrupt_state(self):
        """保存中断状态"""
        try:
            interrupt_file = os.path.join(self.output_dir, "interrupt_state.json")
            interrupt_data = {
                'timestamp': datetime.now().isoformat(),
                'success_count': self.global_status_tracker.success_count,
                'failure_count': self.global_status_tracker.failure_count,
                'system_info': self.resource_monitor.get_system_info(),
                'message': '采样被用户中断'
            }
            with open(interrupt_file, 'w', encoding='utf-8') as f:
                json.dump(interrupt_data, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            print(f"⚠️ 保存中断状态失败: {e}")
    
    def resume_from_interrupt(self) -> bool:
        """从中断点恢复采样"""
        interrupt_file = os.path.join(self.output_dir, "interrupt_state.json")
        if os.path.exists(interrupt_file):
            try:
                with open(interrupt_file, 'r', encoding='utf-8') as f:
                    interrupt_data = json.load(f)
                
                print(f"🔄 检测到中断状态，从 {interrupt_data['timestamp']} 恢复")
                print(f"   中断时成功数: {interrupt_data['success_count']}")
                print(f"   中断时失败数: {interrupt_data['failure_count']}")
                
                # 备份中断文件
                backup_file = os.path.join(self.output_dir, f"interrupt_state_backup_{int(time.time())}.json")
                shutil.copy2(interrupt_file, backup_file)
                
                # 删除中断文件
                os.remove(interrupt_file)
                
                return True
                
            except Exception as e:
                print(f"⚠️ 读取中断状态失败: {e}")
                return False
        
        return False
    
    def generate_dataset(self, target_success_count: int = 2000):
        """Generate the complete dataset with failure handling and progress tracking"""
        print(f"🚀 开始生成数据集，每个模型目标成功数量: {target_success_count}")
        print(f"📁 输出目录: {self.output_dir}")
        print(f"📋 处理文件数量: {len(self.vsp_files)}")
        
        # 检查是否需要从中断点恢复
        if self.resume_from_interrupt():
            print("🔄 从中断点恢复采样...")
        
        total_success_count = 0
        
        for vsp_index, vsp_file in enumerate(self.vsp_files):
            print(f"\n📂 处理文件 {vsp_index + 1}/{len(self.vsp_files)}: {os.path.basename(vsp_file)}")
            
            try:
                # 为每个模型创建独立的状态跟踪器
                model_output_dir = os.path.join(self.output_dir, f"{vsp_index + 1}")
                model_status_tracker = SamplingStatusTracker(model_output_dir)
                model_status_tracker.target_count = target_success_count
                
                # 检查该模型是否已经达到目标数量
                if model_status_tracker.is_complete():
                    print(f"🎉 模型 {os.path.basename(vsp_file)} 已完成，已有 {model_status_tracker.success_count} 个成功采样")
                    total_success_count += model_status_tracker.success_count
                    continue
                
                # Create parameter extractor
                param_extractor = VSPParameterExtractor(vsp_file, model_output_dir)
                
                # Extract parameters
                print("🔍 提取参数...")
                model_params = param_extractor.extract_parameters()
                
                # Create model sampler (现在包含完整的采样和验证流程)
                model_sampler = VSPModelSampler(vsp_file, model_output_dir)
                model_sampler.model_params = model_params  # Share extracted parameters
                model_sampler.status_tracker = model_status_tracker  # 使用模型特定的状态跟踪器
                
                # Generate sample models with immediate validation
                print(f"🎲 开始逐个采样验证（目标: {target_success_count}个成功采样）...")
                sample_paths = model_sampler.sample_models(
                    num_samples=target_success_count, 
                    index=vsp_index, 
                    target_success_count=target_success_count
                )
                
                if not sample_paths:
                    print(f"⚠️ 文件 {os.path.basename(vsp_file)} 没有生成有效采样")
                    continue
                
                print(f"✅ 文件 {os.path.basename(vsp_file)} 处理完成，成功 {len(sample_paths)} 个采样")
                total_success_count += len(sample_paths)
                
            except Exception as e:
                error_type, error_msg = ErrorClassifier.classify_error(e, "文件处理")
                print(f"❌ 处理文件 {os.path.basename(vsp_file)} 失败: {error_msg}")
                self.resource_monitor.cleanup_vsp_state()
        
        # 生成最终报告
        self._generate_final_report(total_success_count, len(self.vsp_files) * target_success_count)
        
        print(f"🎉 数据集生成完成！总共成功生成 {total_success_count} 个采样")
        print(f"📊 平均每个模型: {total_success_count / len(self.vsp_files):.1f} 个成功采样")
    
    def _generate_final_report(self, total_success_count: int, total_target_count: int):
        """生成最终报告"""
        try:
            # 收集所有模型的状态
            all_success_count = 0
            all_failure_count = 0
            all_error_stats = {error_type.value: 0 for error_type in ErrorType}
            
            for vsp_index in range(len(self.vsp_files)):
                model_output_dir = os.path.join(self.output_dir, f"model_{vsp_index + 1}")
                status_file = os.path.join(model_output_dir, "sampling_status.json")
                
                if os.path.exists(status_file):
                    try:
                        with open(status_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            all_success_count += data.get('success_count', 0)
                            all_failure_count += data.get('failure_count', 0)
                            
                            # 合并错误统计
                            error_stats = data.get('error_stats', {})
                            for error_type in ErrorType:
                                if error_type.value in error_stats:
                                    all_error_stats[error_type.value] += error_stats[error_type.value]
                    except Exception as e:
                        print(f"⚠️ 读取模型 {vsp_index + 1} 状态文件失败: {e}")
            
            # 生成总体报告
            report = {
                'total_sampled': all_success_count + all_failure_count,
                'success_count': all_success_count,
                'failure_count': all_failure_count,
                'success_rate': (all_success_count / (all_success_count + all_failure_count) * 100) if (all_success_count + all_failure_count) > 0 else 0,
                'error_distribution': all_error_stats,
                'progress': (all_success_count / total_target_count * 100) if total_target_count > 0 else 0,
                'models_processed': len(self.vsp_files),
                'target_per_model': total_target_count // len(self.vsp_files) if len(self.vsp_files) > 0 else 0
            }
            
            # 保存详细报告
            report_file = os.path.join(self.output_dir, "dataset_report.json")
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)
            
            # 打印摘要
            print(f"\n📊 数据集生成报告:")
            print(f"   总采样数: {report['total_sampled']}")
            print(f"   成功数: {report['success_count']}")
            print(f"   失败数: {report['failure_count']}")
            print(f"   成功率: {report['success_rate']:.2f}%")
            print(f"   进度: {report['progress']:.2f}%")
            print(f"   处理模型数: {report['models_processed']}")
            print(f"   每模型目标: {report['target_per_model']}")
            print(f"   错误分布: {report['error_distribution']}")
            print(f"   详细报告已保存到: {report_file}")
            
        except Exception as e:
            print(f"⚠️ 生成报告失败: {e}")


# Example usage
if __name__ == "__main__":
    target_success_count = 2  # 每个模型的目标成功采样数量

    vsp_dir = "aeromodel"
    vsp_files = [f for f in os.listdir(vsp_dir) if f.endswith('.vsp3')]
    vsp_files.sort(key=lambda fname: int(re.search(r'\d+', fname).group(0)))
    vsp_files = [os.path.join(vsp_dir, f) for f in vsp_files]
    if not vsp_files:
        print(f"No .vsp3 files found in {vsp_dir}")
        exit(1)
        
    print(f"Found {len(vsp_files)} VSP files: {', '.join(os.path.basename(f) for f in vsp_files)}")
    print(f"每个模型目标成功采样数量: {target_success_count}")
    print(f"总目标采样数量: {len(vsp_files) * target_success_count}")
    
    # Generate dataset with improved error handling
    generator = VSPDatasetGenerator(vsp_files, "aero_dataset-1")
    generator.generate_dataset(target_success_count=target_success_count)
