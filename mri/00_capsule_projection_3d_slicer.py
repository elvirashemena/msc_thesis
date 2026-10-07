"""Project the TMS coil target onto the pial surface in 3D Slicer.

Builds a plane from two scalp tangent lines, takes its normal and casts a ray
in both directions from the fiducial marker to find the closest intersection
with the pial surface. The intersection is added as the fiducial
'SurfaceIntersection'.

Usage
-----
1. Install the FreeSurfer extension in 3D Slicer.
2. Load the T1 volume and the pial surface model (node name lh_pial).
3. In Markups, place a point at the centre of the fiducial marker (name F).
4. Draw two scalp tangent lines close to the marker (names L_1 and L_2).
5. Paste this script into the 3D Slicer Python console.

Author: Jakob Litsch litsch@nuuron.com
"""

import vtk
import numpy as np
import slicer

def get_line_vector_and_midpoint(lineNode):
    """Return the direction vector and midpoint of a two-point markup line."""
    p0 = np.array(lineNode.GetNthControlPointPositionWorld(0))
    p1 = np.array(lineNode.GetNthControlPointPositionWorld(1))
    vector = p1 - p0
    midpoint = (p0 + p1) / 2.0
    return vector, midpoint

def project_point_onto_surface(point, normal, surfaceModelNode, visualize=True):
    """Cast a ray both ways along normal from point and return the closest hit.

    Parameters
    ----------
    point : array-like, shape (3,)
        Ray origin in world (RAS) coordinates.
    normal : array-like, shape (3,)
        Ray direction; normalised internally.
    surfaceModelNode : vtkMRMLModelNode
        Surface to intersect.
    visualize : bool
        Add the hit as the fiducial 'SurfaceIntersection' and switch to the
        four-up layout.

    Returns
    -------
    numpy.ndarray or None
        Closest intersection point, or None if the ray misses the surface.
    """
    polyData = surfaceModelNode.GetPolyData()
    if not polyData:
        print("Surface model node has no polydata.")
        return None

    normal = np.array(normal) / np.linalg.norm(normal)
    obbTree = vtk.vtkOBBTree()
    obbTree.SetDataSet(polyData)
    obbTree.BuildLocator()

    p0 = np.array(point)
    ray_length = 1000.0
    all_intersections = []

    for direction in [1.0, -1.0]:
        p1 = p0 + direction * normal * ray_length
        intersectionPoints = vtk.vtkPoints()
        cellIds = vtk.vtkIdList()
        code = obbTree.IntersectWithLine(p0, p1, intersectionPoints, cellIds)
        if code != 0:
            for i in range(intersectionPoints.GetNumberOfPoints()):
                pt = np.array(intersectionPoints.GetPoint(i))
                dist = np.linalg.norm(pt - p0)
                all_intersections.append((pt, dist))

    if not all_intersections:
        print("No intersection found.")
        return None

    closest_point, min_dist = min(all_intersections, key=lambda x: x[1])
    print(f"Intersection at {min_dist:.2f} mm: {closest_point}")

    if visualize:
        markupsNode = slicer.mrmlScene.AddNewNodeByClass(
            'vtkMRMLMarkupsFiducialNode', 'SurfaceIntersection'
        )
        markupsNode.AddControlPoint(np.ravel(closest_point))
        slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
        print("Added 'SurfaceIntersection' fiducial.")

    return closest_point


lineNode1 = slicer.util.getNode('L_1')
lineNode2 = slicer.util.getNode('L_2')

vec1, mid1 = get_line_vector_and_midpoint(lineNode1)
vec2, mid2 = get_line_vector_and_midpoint(lineNode2)

normal = np.cross(vec1, vec2)
if np.linalg.norm(normal) < 1e-6:
    print("Error: Lines are parallel, cannot define unique plane.")
else:
    normal /= np.linalg.norm(normal)
    plane_center = (mid1 + mid2) / 2.0

    print("Plane normal:", normal)
    print("Plane center:", plane_center)

    planeNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsPlaneNode', 'LinesPlane')
    planeNode.SetOrigin(plane_center.tolist())
    planeNode.SetNormal(normal.tolist())
    print("Created plane 'LinesPlane'.")

    modelNode = slicer.util.getNode('lh_pial')
    fiducialNode = slicer.util.getNode('F')

    point = [0.0, 0.0, 0.0]
    fiducialNode.GetNthControlPointPosition(0, point)
    point = np.array(point)

    projectedPoint = project_point_onto_surface(point, normal, modelNode, visualize=True)

    if projectedPoint is not None:
        print("Complete: Plane + projection created!")
        print("Nodes: 'LinesPlane' (plane), 'SurfaceIntersection' (point)")

