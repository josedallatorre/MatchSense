FROM osaiai/dokai:23.05-vpf

RUN python3 -m pip install --no-cache-dir \
    "setuptools<81" wheel scikit-build cmake ninja

COPY requirements.txt requirements.txt

RUN pip3 install --no-cache-dir -r requirements.txt

RUN python3 -m pip install --no-cache-dir \
    --no-build-isolation \
    "git+https://github.com/NVIDIA/VideoProcessingFramework"

RUN python3 -m pip install --no-cache-dir \
    --no-build-isolation \
    "git+https://github.com/NVIDIA/VideoProcessingFramework#subdirectory=src/PytorchNvCodec"

#ENV OPENCV_FFMPEG_CAPTURE_OPTIONS "video_codec;hevc"
