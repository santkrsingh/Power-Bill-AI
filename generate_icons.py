"""
PowerGuard AI — Generate PWA icons (192x192 and 512x512) as PNG files.
Run once: python generate_icons.py
"""
import os, math

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_OK = True
except ImportError:
    PIL_OK = False

def draw_icon(size):
    img = Image.new("RGBA", (size, size), (0,0,0,0))
    d = ImageDraw.Draw(img)

    # Background circle — dark blue
    d.ellipse([0, 0, size-1, size-1], fill="#1e3a5f")

    # Lightning bolt ⚡ drawn as polygon
    cx, cy = size//2, size//2
    s = size * 0.55
    bolt = [
        (cx + s*0.05,  cy - s*0.50),
        (cx - s*0.15,  cy + s*0.05),
        (cx + s*0.08,  cy + s*0.05),
        (cx - s*0.05,  cy + s*0.50),
        (cx + s*0.18,  cy - s*0.02),
        (cx + s*0.00,  cy - s*0.02),
    ]
    bolt = [(int(x), int(y)) for x,y in bolt]
    d.polygon(bolt, fill="#3b82f6")

    # Outer ring accent
    ring_width = max(3, size//40)
    d.ellipse([ring_width, ring_width, size-1-ring_width, size-1-ring_width],
              outline="#3b82f6", width=ring_width)
    return img

if __name__ == "__main__":
    out = os.path.join(os.path.dirname(__file__), "frontend", "icons")
    os.makedirs(out, exist_ok=True)

    if not PIL_OK:
        print("Pillow not installed. Run: pip install Pillow")
        # Create minimal 1x1 placeholder PNGs so the manifest doesn't 404
        import struct, zlib
        def minimal_png(size):
            def pack(*args): return struct.pack(*args)
            def chunk(name, data):
                c = name + data
                return pack('>I', len(data)) + c + pack('>I', zlib.crc32(c) & 0xffffffff)
            sig = b'\x89PNG\r\n\x1a\n'
            ihdr_data = pack('>IIBBBBB', size, size, 8, 2, 0, 0, 0)
            raw = b'\x00' + b'\x00\x4f\x9a' * size  # dark row
            raw_data = zlib.compress(raw * size)
            return sig + chunk(b'IHDR', ihdr_data) + chunk(b'IDAT', raw_data) + chunk(b'IEND', b'')
        for sz in (192, 512):
            with open(os.path.join(out, f"icon-{sz}.png"), "wb") as f:
                f.write(minimal_png(sz))
        print("Minimal placeholder icons created.")
    else:
        for sz in (192, 512):
            img = draw_icon(sz)
            img.save(os.path.join(out, f"icon-{sz}.png"))
            print(f"Created icon-{sz}.png")
        print("Icons saved to frontend/icons/")
