# video-edit

My explorations on video editting and computer vision.

## My playing around

- Yolo Model - does object detection but pretty slow
- Hand Rolled Motion Detector via openCV

## Swing alyzer

I love doing kettlebell swings, but then I realized I was doing them wrong. Instead of fixing my swing form, I wrote an app to analyze them.  Independantly, I fixed my swing form

<iframe width="560" height="315" src="https://www.youtube.com/embed/BD6Sys1GIuU?si=sZLSI7t_8Rs1n1ie" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>


## Other work

[auto-editor](https://github.com/WyattBlue/auto-editor) - Auto trim w/no motion or audio. Except doesn't quite work

     python3 -m auto_editor ~/downloads/delivery.mp4  --ffmpeg_location `which ffmpeg` --edit motion --motion_threshold 0.05
