"""Bilibili Credential 管理器"""

import json
import logging
from pathlib import Path

from bilibili_api import Credential

from .exceptions import CredentialLoadError

logger = logging.getLogger(__name__)


class CredentialManager:
    """Bilibili Credential 统一管理"""

    REQUIRED_FIELDS = ["SESSDATA", "BILI_JCT", "BUVID3", "dedeuserid"]

    @staticmethod
    def load_from_file(cookie_path: Path) -> Credential:
        """从配置文件加载 Credential

        Args:
            cookie_path: Cookie 配置文件路径

        Returns:
            Credential 对象

        Raises:
            CredentialLoadError: 当配置文件不存在、格式错误或缺少必要字段时
        """
        if not cookie_path.exists():
            raise CredentialLoadError(f"Cookie 配置文件不存在: {cookie_path}")

        try:
            with open(cookie_path, encoding="utf-8") as f:
                cookies = json.load(f)
        except json.JSONDecodeError as e:
            raise CredentialLoadError(f"Cookie 配置文件格式错误: {e}")
        except Exception as e:
            raise CredentialLoadError(f"读取 Cookie 配置文件失败: {e}")

        missing_fields = [field for field in CredentialManager.REQUIRED_FIELDS if field not in cookies]
        if missing_fields:
            raise CredentialLoadError(f"Cookie 配置缺少必要字段: {', '.join(missing_fields)}")

        try:
            credential_kwargs = {
                "sessdata": cookies["SESSDATA"],
                "bili_jct": cookies["BILI_JCT"],
                "buvid3": cookies["BUVID3"],
                "dedeuserid": cookies["dedeuserid"],
            }

            if "BUVID4" in cookies:
                credential_kwargs["buvid4"] = cookies["BUVID4"]

            credential = Credential(**credential_kwargs)
            logger.info("Credential 加载成功")
            return credential
        except TypeError as e:
            if "buvid4" in str(e):
                credential_kwargs = {
                    "sessdata": cookies["SESSDATA"],
                    "bili_jct": cookies["BILI_JCT"],
                    "buvid3": cookies["BUVID3"],
                    "dedeuserid": cookies["dedeuserid"],
                }
                credential = Credential(**credential_kwargs)
                logger.info("Credential 加载成功（不含 BUVID4）")
                return credential
            raise CredentialLoadError(f"创建 Credential 失败: {e}")
        except Exception as e:
            raise CredentialLoadError(f"创建 Credential 失败: {e}")
    
    @staticmethod
    def get_cookie_dict(cookie_path: Path) -> dict:
        """获取 Cookie 字典（用于 requests 调用）
        
        Args:
            cookie_path: Cookie 配置文件路径
            
        Returns:
            Cookie 字典
            
        Raises:
            CredentialLoadError: 当配置文件不存在或格式错误时
        """
        if not cookie_path.exists():
            raise CredentialLoadError(f"Cookie 配置文件不存在: {cookie_path}")
        
        try:
            with open(cookie_path, encoding="utf-8") as f:
                cookies = json.load(f)
        except json.JSONDecodeError as e:
            raise CredentialLoadError(f"Cookie 配置文件格式错误: {e}")
        except Exception as e:
            raise CredentialLoadError(f"读取 Cookie 配置文件失败: {e}")
        
        return cookies
