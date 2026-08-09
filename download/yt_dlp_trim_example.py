import yt_dlp
from yt_dlp.utils import download_range_func

def download_video_section(url, start_time, end_time, output_filename="%(title)s_clip.%(ext)s"):
    """
    Downloads a specific section of a YouTube video.
    
    :param url: The YouTube URL (string)
    :param start_time: Start time in seconds (int or float)
    :param end_time: End time in seconds (int or float)
    :param output_filename: The desired output filename template
    """
    
    ydl_opts = {
        # Select best mp4 video and best m4a audio, or fall back to best overall
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        
        # Force the output container to be MP4 when merging video + audio
        'merge_output_format': 'mp4',
        
        # Download only the requested timestamp range
        'download_ranges': download_range_func(None, [(start_time, end_time)]),
        
        # Re-encode keyframes at cut points for exact frame accuracy
        'force_keyframes_at_cuts': True,
        
        # Set output file path
        'outtmpl': output_filename,
    }

    print(f"Downloading section {start_time}s to {end_time}s from: {url}")
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

if __name__ == "__main__":
    # Example: Download from 10 seconds to 25 seconds
    VIDEO_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    
    download_video_section(
        url=VIDEO_URL,
        start_time=10.0, 
        end_time=25.0,
        output_filename="yt_dlp_trim_example.%(ext)s"
    )