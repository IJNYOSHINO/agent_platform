"""
媒体文件验证与处理模块
"""
import base64
import io
from typing import Literal

import filetype
from PIL import Image

# 配置常量
MAX_FILE_SIZE_MB = 5
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

SUPPORTED_IMAGE_MIMES = {"image/jpeg", "image/png", "image/gif", "image/webp", "image/bmp"}
SUPPORTED_AUDIO_MIMES = {"audio/mpeg", "audio/mp3", "audio/wav", "audio/ogg"}
SUPPORTED_PDF_MIMES = {"application/pdf"}

SAFE_IMAGE_FORMAT = "JPEG"
SAFE_IMAGE_MIME = "image/jpeg"


class MediaValidationError(Exception):
    """媒体验证异常"""
    pass


class MediaSanitizer:
    """图片无害化处理器"""

    @classmethod
    def sanitize_image(cls, file_bytes: bytes) -> str:
        """
        图片无害化处理：重新编码为 JPEG
        返回 Base64 编码的 JPEG 图片
        """
        try:
            img = Image.open(io.BytesIO(file_bytes))
            img.verify()
            img = Image.open(io.BytesIO(file_bytes))

            # 转换为 RGB
            img = cls._convert_to_rgb(img)

            # 限制最大尺寸
            img = cls._resize_if_needed(img)

            # 重新编码为 JPEG
            sanitized_bytes = cls._encode_to_jpeg(img)

            return base64.b64encode(sanitized_bytes).decode()

        except Exception as exc:
            raise MediaValidationError(f"Invalid or corrupted image: {exc}") from exc

    @classmethod
    def _convert_to_rgb(cls, img: Image.Image) -> Image.Image:
        """转换为 RGB 模式"""
        if img.mode in ("RGBA", "LA", "P"):
            rgb_img = Image.new("RGB", img.size, (255, 255, 255))
            mask = img.split()[-1] if img.mode == "RGBA" else None
            rgb_img.paste(img, mask=mask)
            return rgb_img
        elif img.mode != "RGB":
            return img.convert("RGB")
        return img

    @classmethod
    def _resize_if_needed(cls, img: Image.Image, max_size: int = 2048) -> Image.Image:
        """如果图片太大，进行缩放"""
        if img.width > max_size or img.height > max_size:
            img.thumbnail((max_size, max_size), Image.LANCZOS)
        return img

    @classmethod
    def _encode_to_jpeg(cls, img: Image.Image, quality: int = 85) -> bytes:
        """编码为 JPEG"""
        output = io.BytesIO()
        img.save(output, format="JPEG", quality=quality, optimize=True)
        return output.getvalue()


class MediaValidator:
    """媒体文件验证器"""

    @classmethod
    def validate_base64(cls, data: str) -> bytes:
        """验证 Base64 格式"""
        try:
            return base64.b64decode(data, validate=True)
        except Exception as exc:
            raise MediaValidationError("Invalid base64 encoding") from exc

    @classmethod
    def validate_file_size(cls, file_bytes: bytes) -> None:
        """验证文件大小"""
        if len(file_bytes) > MAX_FILE_SIZE_BYTES:
            raise MediaValidationError(
                f"File too large: {len(file_bytes) / 1024 / 1024:.2f}MB, "
                f"max {MAX_FILE_SIZE_MB}MB"
            )

    @classmethod
    def detect_mime(cls, file_bytes: bytes) -> str | None:
        """检测真实 MIME 类型"""
        detected = filetype.guess(file_bytes)
        return detected.mime if detected else None

    @classmethod
    def validate_mime(cls, media_type: Literal["image", "audio", "pdf"], mime: str) -> None:
        """验证 MIME 类型是否在支持列表中"""
        if media_type == "image" and mime not in SUPPORTED_IMAGE_MIMES:
            raise MediaValidationError(f"Unsupported image MIME: {mime}")
        if media_type == "audio" and mime not in SUPPORTED_AUDIO_MIMES:
            raise MediaValidationError(f"Unsupported audio MIME: {mime}")
        if media_type == "pdf" and mime not in SUPPORTED_PDF_MIMES:
            raise MediaValidationError(f"Unsupported PDF MIME: {mime}")

    @classmethod
    def validate_pdf(cls, file_bytes: bytes) -> None:
        """验证 PDF 文件头"""
        if not file_bytes.startswith(b'%PDF'):
            raise MediaValidationError("Invalid PDF file: missing PDF header")


class MediaConverter:
    """媒体格式转换器（转 LangChain 格式）"""

    @staticmethod
    def to_langchain_image(data: str) -> dict:
        """转换为 LangChain 图片格式"""
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{SAFE_IMAGE_MIME};base64,{data}"},
        }

    @staticmethod
    def to_langchain_audio(data: str, mime: str) -> dict:
        """转换为 LangChain 音频格式"""
        return {
            "type": "input_audio",
            "input_audio": {
                "data": data,
                "format": mime.split("/")[-1],
            },
        }

    @staticmethod
    def to_langchain_pdf(data: str) -> dict:
        """转换为 LangChain 文本格式（PDF 占位符）"""
        return {
            "type": "text",
            "text": f"[PDF Document, base64 encoded]: {data[:200]}...",
        }