"""
Monkey patch để sửa lỗi compatibility giữa SpeechBrain và huggingface_hub.
Lỗi: SpeechBrain sử dụng 'use_auth_token' nhưng huggingface_hub mới chỉ hỗ trợ 'token'.

Cách sử dụng:
    import fix_speechbrain  # Import TRƯỚC khi import speechbrain
    from speechbrain.inference.speaker import EncoderClassifier
    
Hoặc thêm vào đầu file:
    import fix_speechbrain
"""


def _apply_speechbrain_patch():
    """Áp dụng patch để sửa lỗi use_auth_token."""
    try:
        import speechbrain.utils.fetching as fetching_module
        
        # Kiểm tra xem đã patch chưa
        if hasattr(fetching_module, '_speechbrain_patched'):
            return
        
        # Patch hàm download_file_hf để sửa use_auth_token thành token
        # Đây là nơi hf_kwargs được truyền vào, nên patch ở đây sẽ hoạt động
        original_download_file_hf = fetching_module.download_file_hf
        
        def patched_download_file_hf(hf_kwargs, destination, local_strategy):
            """Patch để sửa use_auth_token thành token."""
            # Sửa use_auth_token thành token nếu có
            if 'use_auth_token' in hf_kwargs:
                hf_kwargs['token'] = hf_kwargs.pop('use_auth_token')
            return original_download_file_hf(hf_kwargs, destination, local_strategy)
        
        fetching_module.download_file_hf = patched_download_file_hf
        fetching_module._speechbrain_patched = True
        
    except ImportError:
        # SpeechBrain chưa được cài đặt, bỏ qua
        pass


# Tự động apply patch khi import module này
# Lưu ý: Phải import TRƯỚC khi import bất kỳ module nào từ speechbrain
_apply_speechbrain_patch()

