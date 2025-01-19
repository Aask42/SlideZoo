import pygame
import sys
import json
import time
import os
import cv2  # For video playback
import hashlib
import requests

# For PDF rendering and creation
try:
    import fitz  # PyMuPDF
except ImportError:
    print("Please install PyMuPDF via: pip install PyMuPDF")
    sys.exit(1)

##############################################################################
#                            HELPER FUNCTIONS
##############################################################################

def load_config(config_path):
    """Load the JSON config file."""
    if not os.path.exists(config_path):
        print(f"Configuration file {config_path} not found.")
        sys.exit(1)
    with open(config_path, 'r') as f:
        return json.load(f)

def save_config(config_path, config):
    """Save the updated config (with AI-generated image paths added)."""
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

def draw_vertical_gradient(surface, color_top, color_bottom):
    """
    Draw a vertical gradient from color_top to color_bottom.
    surface: pygame.Surface
    color_top, color_bottom: (R, G, B)
    """
    width, height = surface.get_size()
    for y in range(height):
        lerp = y / height
        r = int(color_top[0] + (color_bottom[0] - color_top[0]) * lerp)
        g = int(color_top[1] + (color_bottom[1] - color_top[1]) * lerp)
        b = int(color_top[2] + (color_bottom[2] - color_top[2]) * lerp)
        pygame.draw.line(surface, (r, g, b), (0, y), (width, y))

def fade_transition(screen, old_slide_surface, new_slide_surface, duration=1.0):
    """Fade in the new slide over the old slide."""
    clock = pygame.time.Clock()
    start_time = time.time()

    old_surf = old_slide_surface.convert_alpha()
    new_surf = new_slide_surface.convert_alpha()

    while True:
        elapsed = time.time() - start_time
        progress = min(1.0, elapsed / duration)
        alpha = int(progress * 255)

        screen.fill((0, 0, 0))
        old_surf.set_alpha(255)
        screen.blit(old_surf, (0, 0))

        new_surf.set_alpha(alpha)
        screen.blit(new_surf, (0, 0))

        pygame.display.flip()
        clock.tick(60)
        if progress >= 1.0:
            break

def slide_transition(screen, old_slide_surface, new_slide_surface,
                     duration=1.0, direction='left'):
    """Slide old slide out, new slide in."""
    clock = pygame.time.Clock()
    start_time = time.time()
    width, height = screen.get_size()

    old_surf = old_slide_surface.convert_alpha()
    new_surf = new_slide_surface.convert_alpha()

    while True:
        elapsed = time.time() - start_time
        progress = min(1.0, elapsed / duration)

        screen.fill((0, 0, 0))
        if direction == 'left':
            offset_old = int(-width * progress)
            offset_new = offset_old + width
            screen.blit(old_surf, (offset_old, 0))
            screen.blit(new_surf, (offset_new, 0))
        elif direction == 'right':
            offset_old = int(width * progress)
            offset_new = offset_old - width
            screen.blit(old_surf, (offset_old, 0))
            screen.blit(new_surf, (offset_new, 0))
        elif direction == 'up':
            offset_old = int(-height * progress)
            offset_new = offset_old + height
            screen.blit(old_surf, (0, offset_old))
            screen.blit(new_surf, (0, offset_new))
        elif direction == 'down':
            offset_old = int(height * progress)
            offset_new = offset_old - height
            screen.blit(old_surf, (0, offset_old))
            screen.blit(new_surf, (0, offset_new))

        pygame.display.flip()
        clock.tick(60)
        if progress >= 1.0:
            break

def do_transition(screen, old_slide, new_slide, transition_type):
    """
    Decide which transition to use based on 'transition_type' (e.g. 'fade', 'slide').
    """
    if transition_type == "fade":
        fade_transition(screen, old_slide, new_slide, duration=1.0)
    elif transition_type == "slide":
        slide_transition(screen, old_slide, new_slide, duration=1.0, direction='left')
    else:
        # Instant switch if not recognized
        screen.fill((0, 0, 0))
        screen.blit(new_slide, (0, 0))
        pygame.display.flip()

##############################################################################
#   MULTILINE TEXT: MEASURING & RENDERING FOR ALIGNMENT (CENTER or LEFT)
##############################################################################

def measure_multiline_text(text, font, max_width, line_spacing=1.2,
                           bullet_indent=40, bullet_chars=("- ", "* ", "• ")):
    """
    Return (total_width, total_height) of wrapped multiline text
    WITHOUT actually drawing it.
    """
    space_width, space_height = font.size(' ')
    line_height = int(space_height * line_spacing)

    current_x = 0
    current_y = 0
    max_line_width = 0
    all_lines_widths = []

    words = []
    for line in text.split('\n'):
        is_bullet = any(line.strip().startswith(b) for b in bullet_chars)
        bullet_symbol = ""
        if is_bullet:
            for b in bullet_chars:
                if line.strip().startswith(b):
                    bullet_symbol = b.strip()
                    line = line.strip()[len(bullet_symbol):].strip()
                    break
        raw_words = line.split(' ')
        for w in raw_words:
            words.append((w, is_bullet, bullet_symbol))
        words.append(("\n", False, ""))

    bullet_drawn = False

    for word, is_bullet, bullet_symbol in words:
        if word == "\n":
            all_lines_widths.append(current_x)
            max_line_width = max(max_line_width, current_x)
            current_x = 0
            current_y += line_height
            bullet_drawn = False
            continue

        if is_bullet and not bullet_drawn:
            bullet_w, _ = font.size(bullet_symbol)
            current_x += bullet_w + 5
            bullet_drawn = True

        word_w, word_h = font.size(word)
        if current_x + word_w > max_width:
            all_lines_widths.append(current_x)
            max_line_width = max(max_line_width, current_x)
            current_x = 0
            current_y += line_height
            bullet_drawn = False
            if is_bullet:
                bullet_w, _ = font.size(bullet_symbol)
                current_x += bullet_w + 5
                bullet_drawn = True

        current_x += word_w + space_width

    if current_x > 0:
        all_lines_widths.append(current_x)
        max_line_width = max(max_line_width, current_x)

    line_count = len(all_lines_widths)
    total_height = line_count * line_height
    return (max_line_width, total_height)

def draw_multiline_text_aligned(surface, text, font, color, rect,
                                align_center=False, line_spacing=1.5,
                                bullet_indent=40, bullet_chars=("- ", "* ", "• ")):
    """
    Render word-wrapped multiline text, aligned either left or center within rect.
    rect is (x, y, width, height).
    """
    x, y, max_width, max_height = rect
    space_width, space_height = font.size(' ')
    line_height = int(space_height * line_spacing)

    measured_w, measured_h = measure_multiline_text(
        text, font, max_width, line_spacing, bullet_indent, bullet_chars
    )

    if align_center:
        text_block_width = min(measured_w, max_width)
        offset_x = x + (max_width - text_block_width) // 2
    else:
        offset_x = x

    current_x = offset_x
    current_y = y

    words = []
    for line in text.split('\n'):
        is_bullet = any(line.strip().startswith(b) for b in bullet_chars)
        bullet_symbol = ""
        if is_bullet:
            for b in bullet_chars:
                if line.strip().startswith(b):
                    bullet_symbol = b.strip()
                    line = line.strip()[len(bullet_symbol):].strip()
                    break
        raw_words = line.split(' ')
        for w in raw_words:
            words.append((w, is_bullet, bullet_symbol))
        words.append(("\n", False, ""))

    bullet_drawn_on_line = False

    for word, is_bullet, bullet_symbol in words:
        if word == "\n":
            current_x = offset_x
            current_y += line_height
            bullet_drawn_on_line = False
            continue

        if is_bullet and not bullet_drawn_on_line:
            bullet_surface = font.render(bullet_symbol, True, color)
            bullet_rect = bullet_surface.get_rect(topleft=(current_x, current_y))
            surface.blit(bullet_surface, bullet_rect)
            current_x += bullet_rect.width + 5
            bullet_drawn_on_line = True

        word_surface = font.render(word, True, color)
        word_width, word_height = word_surface.get_size()

        if current_x + word_width > offset_x + max_width:
            current_x = offset_x
            current_y += line_height
            bullet_drawn_on_line = False
            if is_bullet:
                bullet_surface = font.render(bullet_symbol, True, color)
                bullet_rect = bullet_surface.get_rect(topleft=(current_x, current_y))
                surface.blit(bullet_surface, bullet_rect)
                current_x += bullet_rect.width + 5
                bullet_drawn_on_line = True

        surface.blit(word_surface, (current_x, current_y))
        current_x += word_width + space_width

##############################################################################
#                           POSITION PARSING
##############################################################################

def parse_position(value, container_dim, content_dim, margin_ratio=0.05):
    """
    Convert a user-specified value for position (x or y) into an absolute pixel
    coordinate. Handles:
        - integer / float
        - "NN%" (percentage)
        - "left"/"top", "center", "right"/"bottom"  (with 5% margin from edges)
    If 'value' is invalid or compound (like "bottom center"), returns 0 as a fallback.
    """

    # If it's already an int/float, just use that
    if isinstance(value, (int, float)):
        return int(value)

    if not isinstance(value, str):
        return 0

    val_str = value.strip().lower()

    # --- Check for multiple align tokens in one string (not supported) ---
    tokens = val_str.split()
    if len(tokens) > 1:
        # e.g. "bottom center"
        print(f"WARNING: parse_position got multiple align tokens '{value}'. "
              f"Use 'x': 'center', 'y': 'bottom' instead. Defaulting to 0.")
        return 0

    # If exactly 1 token, proceed
    margin_pixels = int(container_dim * margin_ratio)  # e.g. 5% margin

    # Percentage
    if val_str.endswith('%'):
        try:
            perc = float(val_str.strip('%'))
            return int((container_dim - content_dim) * (perc / 100.0))
        except ValueError:
            return 0

    if val_str in ('left', 'top'):
        return margin_pixels
    elif val_str in ('center', 'centre'):
        return (container_dim - content_dim) // 2
    elif val_str in ('right', 'bottom'):
        return (container_dim - content_dim - margin_pixels)

    # Otherwise, try integer parse
    try:
        return int(val_str)
    except ValueError:
        return 0

##############################################################################
#                       OPENAI IMAGE GENERATION (Direct POST)
##############################################################################

def call_openai_image_api(prompt, api_key, n=1, size="512x512"):
    """
    Calls the v1/images/generations endpoint directly with `requests`.
    Returns the first image URL from the response.
    """
    url = "https://api.openai.com/v1/images/generations"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "prompt": prompt,
        "n": n,
        "size": size
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code != 200:
        raise Exception(f"OpenAI API error {response.status_code}: {response.text}")
    resp_json = response.json()
    return resp_json['data'][0]['url']  # The first image's URL

def generate_openai_images_for_slides(config_path, config):
    """
    For each slide in config['slides'], if "auto_openai_image" = true, then:
      1. Construct a prompt from the slide or use 'openai_prompt'.
      2. Check if we have a cached file for that prompt (via an MD5-based filename).
      3. If not cached, call OpenAI to generate the image, save to 'cache/'.
      4. Insert an entry into slide["extra_images"] with path to the PNG,
         placed in the middle right by default.
      5. Write updated config back to disk so next run we won't re-generate.
    """
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("OPENAI_API_KEY is not set. Skipping AI image generation.")
        return config  # no changes

    if not os.path.exists("cache"):
        os.makedirs("cache")

    slides = config.get("slides", [])
    changed_config = False

    for idx, slide in enumerate(slides):
        if slide.get("auto_openai_image", False) is not True:
            continue  # Not enabled for this slide

        user_prompt = slide.get("openai_prompt", None)
        if not user_prompt:
            title = slide.get("title", "")
            content = slide.get("content", "")
            user_prompt = (
                f"Generate an image for a presentation slide titled '{title}'. "
                f"Content: {content}"
            )

        prompt_hash = hashlib.md5(user_prompt.encode("utf-8")).hexdigest()[:10]
        cache_filename = f"cache/slide_{idx}_{prompt_hash}.png"

        if not os.path.exists(cache_filename):
            print(f"[AI] Generating image for slide #{idx} with prompt:\n   {user_prompt}")
            try:
                image_url = call_openai_image_api(user_prompt, openai_api_key, size="512x512", n=1)
                print(f"[AI] Image URL: {image_url}")

                # Download
                img_data = requests.get(image_url).content
                with open(cache_filename, 'wb') as f:
                    f.write(img_data)
                print(f"[AI] Saved to {cache_filename}")
            except Exception as e:
                print(f"[AI] Error generating image for slide #{idx}: {e}")
                continue

        extra_imgs = slide.setdefault("extra_images", [])
        already_listed = any(
            (img.get("path") == cache_filename) for img in extra_imgs
        )
        if not already_listed:
            # Place it in the "middle right"
            extra_imgs.append({
                "path": cache_filename,
                "x": "right",    # far right
                "y": "center",   # vertically centered
                "max_width": 512,
                "max_height": 512,
                "keep_aspect": True
            })
            print(f"[AI] Inserted {cache_filename} into slide #{idx}'s extra_images.")
            changed_config = True

    if changed_config:
        print("[AI] Updating config file with newly generated image paths...")
        save_config(config_path, config)

    return config

##############################################################################
#                  RENDER SLIDES (TEXT / VIDEO / PDF)
##############################################################################

import io

def render_pdf_pages_to_surface(pdf_path, page_zoom=1.0):
    """
    Load a PDF file via PyMuPDF, render each page, and stack them vertically
    into one tall Pygame Surface. Return that surface.
    """
    doc = fitz.open(pdf_path)
    page_surfaces = []

    for page_index in range(len(doc)):
        page = doc.load_page(page_index)
        mat = fitz.Matrix(page_zoom, page_zoom)
        pix = page.get_pixmap(matrix=mat)
        # Convert to PNG bytes
        img_bytes = pix.tobytes(output="png")
        # Make a pygame Surface
        if pix.alpha:
            page_surface = pygame.image.load_extended(img_bytes).convert_alpha()
        else:
            page_surface = pygame.image.load_extended(img_bytes).convert()
        page_surfaces.append(page_surface)

    doc.close()

    if not page_surfaces:
        return pygame.Surface((800, 600))

    total_width = max(surf.get_width() for surf in page_surfaces)
    total_height = sum(surf.get_height() for surf in page_surfaces)

    combined_surface = pygame.Surface((total_width, total_height), pygame.SRCALPHA).convert_alpha()
    y_offset = 0
    for psurf in page_surfaces:
        combined_surface.blit(psurf, (0, y_offset))
        y_offset += psurf.get_height()

    return combined_surface

def get_first_pdf_page_surface(pdf_path, width, height):
    """
    Loads only the FIRST page of the PDF, scaled to fit (width, height).
    Returns a single Surface for the placeholder.
    """
    doc = fitz.open(pdf_path)
    if len(doc) == 0:
        doc.close()
        return pygame.Surface((width, height))

    page = doc.load_page(0)
    pix = page.get_pixmap()
    img_bytes = pix.tobytes(output="png")
    doc.close()

    surf = pygame.image.load_extended(img_bytes).convert_alpha()
    # scale it to fit
    final_surf = pygame.transform.smoothscale(surf, (width, height))
    return final_surf

def render_pdf_slide(config, width, height, slide_data, font_title, font_subtitle):
    """
    Create a "placeholder" surface for the PDF background,
    plus we can show the first PDF page scaled to fit, if we want.
    """
    slide_surface = pygame.Surface((width, height), pygame.SRCALPHA).convert_alpha()

    bg_style = slide_data.get("bg_style", "solid").lower()
    if bg_style == "gradient":
        color_top = slide_data.get("bg_color_top", [100, 100, 100])
        color_bottom = slide_data.get("bg_color_bottom", [50, 50, 50])
        draw_vertical_gradient(slide_surface, color_top, color_bottom)
    else:
        bg_color = slide_data.get("bg_color", [0, 0, 0])
        slide_surface.fill(bg_color)

    font_color = slide_data.get("font_color", [255, 255, 255])

    title_text = slide_data.get("title", "")
    if title_text:
        title_surface = font_title.render(title_text, True, font_color)
        title_rect = title_surface.get_rect(center=(width // 2, height // 10))
        slide_surface.blit(title_surface, title_rect)

    subtitle_text = slide_data.get("subtitle", "")
    if subtitle_text:
        subtitle_surface = font_subtitle.render(subtitle_text, True, font_color)
        subtitle_rect = subtitle_surface.get_rect(center=(width // 2, height // 5))
        slide_surface.blit(subtitle_surface, subtitle_rect)

    pdf_path = slide_data.get("pdf_file", "")
    if os.path.exists(pdf_path):
        # Render the first page as a "placeholder"
        first_page_surf = get_first_pdf_page_surface(pdf_path, width//2, height//2)
        # place it centered
        x_pos = (width - first_page_surf.get_width()) // 2
        y_pos = (height - first_page_surf.get_height()) // 2
        slide_surface.blit(first_page_surf, (x_pos, y_pos))

    return slide_surface

def get_first_frame_of_video(video_path):
    """
    Return a single frame (as a pygame.Surface) from the video, or None if fail.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
    return surface

def render_video_slide(config, width, height, slide_data, font_title, font_subtitle):
    """
    Render a placeholder surface for video slides (with gradient or solid background).
    We also place the FIRST frame in the center if we can get it.
    """
    slide_surface = pygame.Surface((width, height), pygame.SRCALPHA).convert_alpha()

    bg_style = slide_data.get("bg_style", "solid").lower()
    if bg_style == "gradient":
        color_top = slide_data.get("bg_color_top", [100, 100, 100])
        color_bottom = slide_data.get("bg_color_bottom", [50, 50, 50])
        draw_vertical_gradient(slide_surface, color_top, color_bottom)
    else:
        bg_color = slide_data.get("bg_color", [0, 0, 0])
        slide_surface.fill(bg_color)

    font_color = slide_data.get("font_color", [255, 255, 255])

    title_text = slide_data.get("title", "")
    if title_text:
        title_surface = font_title.render(title_text, True, font_color)
        title_rect = title_surface.get_rect(center=(width // 2, height // 10))
        slide_surface.blit(title_surface, title_rect)

    subtitle_text = slide_data.get("subtitle", "")
    if subtitle_text:
        subtitle_surface = font_subtitle.render(subtitle_text, True, font_color)
        subtitle_rect = subtitle_surface.get_rect(center=(width // 2, height // 5))
        slide_surface.blit(subtitle_surface, subtitle_rect)

    video_path = slide_data.get("video_file", "")
    if os.path.exists(video_path):
        frame_surf = get_first_frame_of_video(video_path)
        if frame_surf:
            # letterbox it to fit width x height
            frame_w, frame_h = frame_surf.get_size()
            aspect_vid = frame_w / frame_h
            aspect_screen = width / height
            if aspect_screen > aspect_vid:
                new_height = height // 2
                new_width = int(new_height * aspect_vid)
            else:
                new_width = width // 2
                new_height = int(new_width / aspect_vid)
            frame_surf = pygame.transform.smoothscale(frame_surf, (new_width, new_height))
            x_pos = (width - new_width) // 2
            y_pos = (height - new_height) // 2
            slide_surface.blit(frame_surf, (x_pos, y_pos))

    return slide_surface

def render_text_slide(config, width, height, slide_data, font_title, font_subtitle):
    """
    Render a text/image slide with optional 'extra_images'.
    If `auto_openai_image` is True, we do special layout:
        - The AI image is on the far right (x="right", y="center").
        - The text is placed in the horizontal space between the left margin
          and the left edge of the image, centered horizontally and vertically.
    Otherwise, fallback to normal logic.
    """
    slide_surface = pygame.Surface((width, height), pygame.SRCALPHA).convert_alpha()

    bg_style = slide_data.get("bg_style", "solid").lower()
    if bg_style == "gradient":
        color_top = slide_data.get("bg_color_top", [100, 100, 100])
        color_bottom = slide_data.get("bg_color_bottom", [50, 50, 50])
        draw_vertical_gradient(slide_surface, color_top, color_bottom)
    else:
        bg_color = slide_data.get("bg_color", [255, 255, 255])
        slide_surface.fill(bg_color)

    font_color = slide_data.get("font_color", [0, 0, 0])

    # Title & subtitle
    title_text = slide_data.get("title", "")
    if title_text:
        title_surface = font_title.render(title_text, True, font_color)
        title_rect = title_surface.get_rect(center=(width // 2, height // 8))
        slide_surface.blit(title_surface, title_rect)

    subtitle_text = slide_data.get("subtitle", "")
    if subtitle_text:
        subtitle_surface = font_subtitle.render(subtitle_text, True, font_color)
        subtitle_rect = subtitle_surface.get_rect(center=(width // 2, height // 5))
        slide_surface.blit(subtitle_surface, subtitle_rect)

    # Load & blit images
    extra_imgs = slide_data.get("extra_images", [])
    ai_image_rect = None
    for eimg in extra_imgs:
        img_path = eimg.get("path", None)
        if not img_path or not os.path.exists(img_path):
            continue
        pic = pygame.image.load(img_path).convert_alpha()
        pic_rect = pic.get_rect()

        max_w = eimg.get("max_width", pic_rect.width)
        max_h = eimg.get("max_height", pic_rect.height)
        keep_aspect = eimg.get("keep_aspect", False)

        if keep_aspect:
            aspect = pic_rect.width / pic_rect.height
            final_w = min(pic_rect.width, max_w)
            final_h = min(pic_rect.height, max_h)
            scaled_h = int(final_w / aspect)
            if scaled_h > final_h:
                scaled_h = final_h
                final_w = int(scaled_h * aspect)
            final_size = (final_w, scaled_h)
        else:
            final_w = min(pic_rect.width, max_w)
            final_h = min(pic_rect.height, max_h)
            final_size = (final_w, final_h)

        pic = pygame.transform.smoothscale(pic, final_size)

        x_pos = parse_position(eimg.get("x", 0), width, final_w)
        y_pos = parse_position(eimg.get("y", 0), height, final_h)

        slide_surface.blit(pic, (x_pos, y_pos))

        if slide_data.get("auto_openai_image", False) is True and eimg.get("x") == "right":
            ai_image_rect = pygame.Rect(x_pos, y_pos, final_w, final_h)

    # Place the text
    content_text = slide_data.get("content", "")
    if content_text:
        body_font_size = config.get("font_size_body", 36)
        body_font = pygame.font.Font(config.get("font_name", "freesansbold.ttf"), body_font_size)

        if slide_data.get("auto_openai_image", False) is True and ai_image_rect is not None:
            left_margin = 50
            right_margin_from_image = 50
            text_area_width = ai_image_rect.x - right_margin_from_image - left_margin
            if text_area_width < 100:
                text_area_width = width - (left_margin * 2)

            measured_w, measured_h = measure_multiline_text(
                content_text, body_font, text_area_width, line_spacing=1.2
            )
            region_x = left_margin
            region_w = text_area_width
            text_x = region_x + (region_w - measured_w) // 2
            text_y = (height - measured_h) // 2
            if text_x < 0:
                text_x = 0

            text_rect = (text_x, text_y, text_area_width, measured_h)
            draw_multiline_text_aligned(
                slide_surface,
                content_text,
                body_font,
                font_color,
                text_rect,
                align_center=False,
                line_spacing=1.2
            )
        else:
            # Original fallback logic
            content_top = int(height * 0.35)
            content_left = 50
            content_right_margin = 50
            total_text_width = width - (content_left + content_right_margin)
            content_height = height - content_top - 50

            text_rect = (content_left, content_top, total_text_width, content_height)
            draw_multiline_text_aligned(
                slide_surface,
                content_text,
                body_font,
                font_color,
                text_rect,
                align_center=True,
                line_spacing=1.2
            )

    return slide_surface

##############################################################################
#                   PDF VIEWER (SCROLLABLE)
##############################################################################

def run_pdf_viewer(screen, slide_data):
    """
    Display the PDF with vertical scrolling in a loop.
    - W / S or PageUp / PageDown: scroll up/down
    - Left/Right or Up/Down keys: go prev/next slide
    - Mouse click: next slide
    - ESC: quit
    Returns: 'next', 'prev', 'quit', or 'done'
    """
    pdf_path = slide_data.get("pdf_file", "")
    if not os.path.exists(pdf_path):
        print(f"Could not find PDF file: {pdf_path}")
        return 'done'

    try:
        pdf_surface = render_pdf_pages_to_surface(pdf_path, page_zoom=1.0)
    except Exception as e:
        print(f"Error rendering PDF: {e}")
        return 'done'

    scroll_y = 0
    scroll_step = 50
    clock = pygame.time.Clock()

    while True:
        screen_width, screen_height = screen.get_size()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return 'quit'
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return 'quit'
                elif event.key in (pygame.K_LEFT, pygame.K_DOWN):
                    return 'prev'
                elif event.key in (pygame.K_RIGHT, pygame.K_UP):
                    return 'next'
                elif event.key in (pygame.K_w, pygame.K_PAGEUP):
                    scroll_y = max(0, scroll_y - scroll_step)
                elif event.key in (pygame.K_s, pygame.K_PAGEDOWN):
                    max_scroll = max(0, pdf_surface.get_height() - screen_height)
                    scroll_y = min(max_scroll, scroll_y + scroll_step)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                return 'next'
            elif event.type == pygame.VIDEORESIZE:
                screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)

        screen.fill((0, 0, 0))
        screen.blit(pdf_surface, (0, -scroll_y))

        max_scroll = max(0, pdf_surface.get_height() - screen_height)
        if max_scroll > 0:
            scrollbar_height = int(screen_height * (screen_height / pdf_surface.get_height()))
            scrollbar_y = int((scroll_y / max_scroll) * (screen_height - scrollbar_height))
            pygame.draw.rect(screen, (80, 80, 80), (screen_width - 10, 0, 10, screen_height))
            pygame.draw.rect(screen, (200, 200, 200), (screen_width - 10, scrollbar_y, 10, scrollbar_height))

        pygame.display.flip()
        clock.tick(60)

    return 'done'

##############################################################################
#                   VIDEO PLAYER (LETTERBOX, SKIPPABLE)
##############################################################################

def run_video_player(screen, slide_data):
    """
    Plays the video with letterbox/pillarbox by default.
    If 'video_x' or 'video_y' are set, we use parse_position to place it.
    Skip if user clicks or presses RIGHT.
    Return 'next', 'prev', 'quit', or 'done'.
    """
    video_path = slide_data.get("video_file", "")
    if not os.path.exists(video_path):
        print(f"Could not find video file: {video_path}")
        return 'done'

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video file: {video_path}")
        return 'done'

    clock = pygame.time.Clock()
    paused = False

    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30

    while True:
        screen_width, screen_height = screen.get_size()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                cap.release()
                return 'quit'
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    cap.release()
                    return 'quit'
                elif event.key == pygame.K_LEFT or event.key == pygame.K_DOWN:
                    cap.release()
                    return 'prev'
                elif event.key == pygame.K_RIGHT or event.key == pygame.K_UP:
                    cap.release()
                    return 'next'
                elif event.key == pygame.K_SPACE or event.key == pygame.K_TAB:
                    paused = not paused
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # skip
                cap.release()
                return 'next'
            elif event.type == pygame.VIDEORESIZE:
                screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)

        if not paused:
            ret, frame = cap.read()
            if not ret:
                # End of video
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            aspect_vid = vid_w / vid_h
            aspect_screen = screen_width / screen_height

            if aspect_screen > aspect_vid:
                new_height = screen_height
                new_width = int(new_height * aspect_vid)
            else:
                new_width = screen_width
                new_height = int(new_width / aspect_vid)

            offset_x = (screen_width - new_width) // 2
            offset_y = (screen_height - new_height) // 2

            video_x = slide_data.get("video_x", None)
            video_y = slide_data.get("video_y", None)
            if video_x is not None:
                offset_x = parse_position(video_x, screen_width, new_width)
            if video_y is not None:
                offset_y = parse_position(video_y, screen_height, new_height)

            frame_surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
            frame_surface = pygame.transform.smoothscale(frame_surface, (new_width, new_height))

            screen.fill((0, 0, 0))
            screen.blit(frame_surface, (offset_x, offset_y))

            # progress bar
            current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            if total_frames > 0:
                progress_fraction = current_frame / total_frames
            else:
                progress_fraction = 0.0

            bar_height = 8
            bar_bg_color = (50, 50, 50)
            bar_color = (200, 0, 0)

            pygame.draw.rect(
                screen,
                bar_bg_color,
                (0, screen_height - bar_height, screen_width, bar_height)
            )
            pygame.draw.rect(
                screen,
                bar_color,
                (0, screen_height - bar_height,
                 int(screen_width * progress_fraction), bar_height)
            )

            pygame.display.flip()

        clock.tick(fps)

    cap.release()
    return 'done'

##############################################################################
#                     EXPORT ALL SLIDES TO A SINGLE PDF
##############################################################################

def render_slide_surface_for_pdf(config, slide_data, width, height,
                                 font_title, font_subtitle):
    """
    Similar to our "render_text_slide"/"render_video_slide"/"render_pdf_slide"
    but returns a single *static* snapshot.
      - text slides => normal
      - pdf slides  => first page
      - video slides => first frame
    """
    slide_type = slide_data.get("slide_type", "text").lower()

    if slide_type == "video":
        return render_video_slide(config, width, height, slide_data, font_title, font_subtitle)
    elif slide_type == "pdf":
        return render_pdf_slide(config, width, height, slide_data, font_title, font_subtitle)
    else:
        # text or anything else
        return render_text_slide(config, width, height, slide_data, font_title, font_subtitle)

def export_slides_to_pdf(config, slides_data, output_pdf="exported_slides.pdf"):
    """
    Generates a single static PDF with one page per slide.
    For 'video' slides, we grab the first frame. For 'pdf' slides, we show the first page, etc.
    """
    print(f"Exporting {len(slides_data)} slides to PDF: {output_pdf}")

    doc = fitz.open()

    width_px = config.get("width", 800)
    height_px = config.get("height", 600)

    font_name = config.get("font_name", "freesansbold.ttf")
    font_size_title = config.get("font_size_title", 36)
    font_size_subtitle = config.get("font_size_subtitle", 24)
    font_title = pygame.font.Font(font_name, font_size_title)
    font_subtitle = pygame.font.Font(font_name, font_size_subtitle)

    for idx, slide_data in enumerate(slides_data):
        # 1) Render the slide to a Pygame surface
        slide_surface = render_slide_surface_for_pdf(
            config, slide_data, width_px, height_px,
            font_title, font_subtitle
        )
        # 2) Convert that surface to PNG bytes in memory
        import io
        temp_buffer = io.BytesIO()
        # use 'save_extended' to ensure PNG format
        pygame.image.save_extended(slide_surface, temp_buffer, ".png")

        # 3) Make a new page in the PDF doc
        page = doc.new_page(width=width_px, height=height_px)

        # 4) Insert image from memory
        rect = fitz.Rect(0, 0, width_px, height_px)
        page.insert_image(rect, stream=temp_buffer.getvalue())

    doc.save(output_pdf)
    doc.close()
    print(f"PDF export complete: {output_pdf}")

##############################################################################
#                                MAIN LOOP
##############################################################################

def run_slideshow(config_path='config.json'):
    pygame.init()

    screen_flags = pygame.RESIZABLE
    config = load_config(config_path)

    # Possibly generate images for slides & update config
    config = generate_openai_images_for_slides(config_path, config)

    initial_width = config.get("width", 800)
    initial_height = config.get("height", 600)
    screen = pygame.display.set_mode((initial_width, initial_height), screen_flags)
    pygame.display.set_caption(config.get("window_title", "Snazzy Slideshow"))

    font_name = config.get("font_name", "freesansbold.ttf")
    font_size_title = config.get("font_size_title", 36)
    font_size_subtitle = config.get("font_size_subtitle", 24)
    font_title = pygame.font.Font(font_name, font_size_title)
    font_subtitle = pygame.font.Font(font_name, font_size_subtitle)

    slides_data = config.get("slides", [])
    if not slides_data:
        print("No slides found in configuration.")
        pygame.quit()
        return

    def render_all_slides(slide_list, w, h):
        s_surfs = []
        for sdata in slide_list:
            stype = sdata.get("slide_type", "text").lower()
            if stype == "video":
                s_surfs.append(render_video_slide(config, w, h, sdata, font_title, font_subtitle))
            elif stype == "pdf":
                s_surfs.append(render_pdf_slide(config, w, h, sdata, font_title, font_subtitle))
            else:
                s_surfs.append(render_text_slide(config, w, h, sdata, font_title, font_subtitle))
        return s_surfs

    current_slides = render_all_slides(slides_data, initial_width, initial_height)
    current_index = 0
    total_slides = len(current_slides)

    screen.fill((0, 0, 0))
    screen.blit(current_slides[current_index], (0, 0))
    pygame.display.flip()

    clock = pygame.time.Clock()
    running = True

    while running:
        slide_data = slides_data[current_index]
        slide_type = slide_data.get("slide_type", "text").lower()

        if slide_type == "video":
            result = run_video_player(screen, slide_data)
            if result == 'quit':
                running = False
            elif result == 'next':
                old_slide = current_slides[current_index]
                current_index = (current_index + 1) % total_slides
                new_slide = current_slides[current_index]
                do_transition(
                    screen, old_slide, new_slide,
                    slides_data[current_index].get("transition", "fade")
                )
            elif result == 'prev':
                old_slide = current_slides[current_index]
                current_index = (current_index - 1) % total_slides
                new_slide = current_slides[current_index]
                do_transition(
                    screen, old_slide, new_slide,
                    slides_data[current_index].get("transition", "fade")
                )
            else:
                old_slide = current_slides[current_index]
                current_index = (current_index + 1) % total_slides
                new_slide = current_slides[current_index]
                do_transition(
                    screen, old_slide, new_slide,
                    slides_data[current_index].get("transition", "fade")
                )

        elif slide_type == "pdf":
            result = run_pdf_viewer(screen, slide_data)
            if result == 'quit':
                running = False
            elif result == 'next':
                old_slide = current_slides[current_index]
                current_index = (current_index + 1) % total_slides
                new_slide = current_slides[current_index]
                do_transition(
                    screen, old_slide, new_slide,
                    slides_data[current_index].get("transition", "fade")
                )
            elif result == 'prev':
                old_slide = current_slides[current_index]
                current_index = (current_index - 1) % total_slides
                new_slide = current_slides[current_index]
                do_transition(
                    screen, old_slide, new_slide,
                    slides_data[current_index].get("transition", "fade")
                )
            else:
                old_slide = current_slides[current_index]
                current_index = (current_index + 1) % total_slides
                new_slide = current_slides[current_index]
                do_transition(
                    screen, old_slide, new_slide,
                    slides_data[current_index].get("transition", "fade")
                )

        else:  # "text"
            slide_active = True
            while slide_active:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        slide_active = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                            slide_active = False
                        elif event.key in (pygame.K_RIGHT, pygame.K_UP):
                            old_slide = current_slides[current_index]
                            current_index = (current_index + 1) % total_slides
                            new_slide = current_slides[current_index]
                            do_transition(
                                screen, old_slide, new_slide,
                                slides_data[current_index].get("transition", "fade")
                            )
                            slide_active = False
                        elif event.key in (pygame.K_LEFT, pygame.K_DOWN):
                            old_slide = current_slides[current_index]
                            current_index = (current_index - 1) % total_slides
                            new_slide = current_slides[current_index]
                            do_transition(
                                screen, old_slide, new_slide,
                                slides_data[current_index].get("transition", "fade")
                            )
                            slide_active = False
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        # same as RIGHT arrow
                        old_slide = current_slides[current_index]
                        current_index = (current_index + 1) % total_slides
                        new_slide = current_slides[current_index]
                        do_transition(
                            screen, old_slide, new_slide,
                            slides_data[current_index].get("transition", "fade")
                        )
                        slide_active = False
                    elif event.type == pygame.VIDEORESIZE:
                        new_width, new_height = event.w, event.h
                        screen = pygame.display.set_mode((new_width, new_height), pygame.RESIZABLE)
                        current_slides = render_all_slides(slides_data, new_width, new_height)
                        screen.fill((0,0,0))
                        screen.blit(current_slides[current_index], (0, 0))
                        pygame.display.flip()

                clock.tick(60)

        if not running:
            break

    pygame.quit()

    # After the slideshow ends, export to PDF
    out_pdf_path = config.get("output_pdf", "exported_slides.pdf")
    export_slides_to_pdf(config, slides_data, out_pdf_path)

if __name__ == "__main__":
    run_slideshow("config.json")
