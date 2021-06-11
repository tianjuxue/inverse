# import gmsh
# import sys
# import math

# # Init GMSH
# gmsh.initialize(sys.argv)
# # Ask GMSH to display information in the terminal
# gmsh.option.setNumber("General.Terminal", 1)

# # Create a model and name it "MyCircle"
# model = gmsh.model
# model.add("MyCircle")

# # Parameters
# R1 = 1  # Radius
# h = 0.1 # Mesh size

# # Create Point for the center of the circle
# center = model.geo.addPoint(0,0,0, h, 10)
# # Create 3 Points on the circle
# points = []
# for j in range(3):
#   points.append(model.geo.addPoint(R1*math.cos(2*math.pi*j/3), R1*math.sin(2*math.pi*j/3), 0, h))
# # Create 3 circle arc
# lines = []
# for j in range(3):
#   lines.append(model.geo.addCircleArc(points[j],center,points[(j+1)%3]))
# # Curveloop and Surface
# curveloop = model.geo.addCurveLoop([1,2,3])
# disk = model.geo.addPlaneSurface([curveloop])

# # Physical groups
# # gmsh.model.addPhysicalGroup(dim, list of tags, physical tag)
# gmsh.model.addPhysicalGroup(1, lines, 1)
# gmsh.model.addPhysicalGroup(2, [disk], 10)

# # This command is mandatory and synchronize CAD with GMSH Model. The less you launch it, the better it is for performance purpose
# gmsh.model.geo.synchronize()
# # Mesh (2D)
# model.mesh.generate(2)
# # Write on disk
# gmsh.write("MyDisk.msh")
# # Launch the GUI (not mandatory at all)
# gmsh.fltk.run();
# # Finalize GMSH
# gmsh.finalize()
import gmsh
import sys
import math

# Init GMSH
gmsh.initialize(sys.argv)
# Ask GMSH to display information in the terminal
gmsh.option.setNumber("General.Terminal", 1)

# Create a model and name it "MyCircle"
model = gmsh.model
model.add("MyCircle")

# Parameters
R1 = 1  # Radius
h = 0.1 # Mesh size

# Create Point for the center of the circle
center = model.geo.addPoint(0,0,0, h, 10)
# Create 3 Points on the circle
points = []
for j in range(3):
  points.append(model.geo.addPoint(R1*math.cos(2*math.pi*j/3), R1*math.sin(2*math.pi*j/3), 0, h))
# Create 3 circle arc
lines = []
for j in range(3):
  lines.append(model.geo.addCircleArc(points[j],center,points[(j+1)%3]))
# Curveloop and Surface
curveloop = model.geo.addCurveLoop([1,2,3])
disk = model.geo.addPlaneSurface([curveloop])

# Physical groups
# gmsh.model.addPhysicalGroup(dim, list of tags, physical tag)
gmsh.model.addPhysicalGroup(1, lines, 1)
gmsh.model.addPhysicalGroup(2, [disk], 10)

# This command is mandatory and synchronize CAD with GMSH Model. The less you launch it, the better it is for performance purpose
gmsh.model.geo.synchronize()
# Mesh (2D)
model.mesh.generate(2)
# Write on disk
gmsh.write("MyDisk.msh")
# Launch the GUI (not mandatory at all)
gmsh.fltk.run();
# Finalize GMSH
gmsh.finalize()