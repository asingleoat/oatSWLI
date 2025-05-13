// === OpenSCAD Heightmap Loader ===

// Parameters
scale_factor = 1;  // Scale in X and Y directions
z_scale = 1;  // Scale in the Z direction (height exaggeration)


// Load heightmap from an image (grayscale PNG/JPG)
heightmap = "out.png";  // ✅ Change this to your image filename

// Create 3D terrain from the image
translate([0, 0, 0])
// rotate([90,0,0])
scale([scale_factor, scale_factor, z_scale])
surface(heightmap, center=true, convexity=5);
