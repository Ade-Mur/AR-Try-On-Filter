# How to Add a Hat to Your AR Project

## Quick Start

1. **Find a hat image** (PNG format with transparency)
2. **Name it `hat.png`**
3. **Place it in your project folder**
4. **Run the complete AR overlay:**

```powershell
py -3.12 complete_ar_overlay.py
```

## Image Requirements

### Format
- **Must be PNG** with alpha channel (transparency)
- Recommended size: 1000x500 to 4000x2000 pixels

### Design Tips
- **Hat should face left** (as seen in image)
- **Transparent background** around the hat
- **High resolution** for best quality
- **Centered in the image** if possible

### Where to Get Hat Images

#### Free Resources:
- **Pinterest** - Search "hat PNG transparent"
- **Pixabay** (pixabay.com) - Free PNG images
- **Pexels** (pexels.com) - Stock photos
- **Freepik** (freepik.com) - Free vector graphics
- **FlatIcon** (flaticon.com) - Icon-based hats
- **Canva** - Design your own hat

#### Create Your Own:
- Use **Canva** or **Photoshop**
- Design a hat on transparent background
- Export as PNG
- Make sure hat is facing left

## Examples

Good hat images:
- Baseball cap (facing left)
- Beanie (winter hat)
- Top hat
- Cowboy hat
- Snapback

## Technical Details

The hat positioning uses:
- **Forehead landmarks** - Detects top of your head
- **Head width calculation** - Scales to fit your head
- **Vertical offset** - Positions above forehead
- **Alpha blending** - Smooth transparency

## Troubleshooting

### Hat too small/large?
Edit `complete_ar_overlay.py` line ~202:
```python
head_width * 1.3  # Change 1.3 to adjust size
```

### Hat positioned wrong?
Edit line ~210:
```python
vertical_offset = -int(eye_distance * 1.2)  # Change 1.2 for height
```

### Hat not showing?
- Check hat.png is in project folder
- Verify it's PNG format with alpha channel
- Make sure image loads: Check console messages

## Enjoy Your AR Hat!

The hat will automatically track your head movement and rotate naturally with your face! 🎩

