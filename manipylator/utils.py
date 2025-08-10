import numpy as np
import os
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from scipy.spatial.transform import Rotation as R


def parametric_heart_1(t):
    x = np.sqrt(2) * np.sin(t) ** 3
    y = -(np.cos(t) ** 3) - np.cos(t) ** 2 + 2 * np.cos(t)

    return np.stack((x, y, np.zeros(t.shape)), axis=1)


def parametric_heart_2(t):
    x = (16 * np.sin(t) ** 3) / 11
    y = (
        13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - 1 * np.cos(4 * t) - 5
    ) / 11

    return np.stack((x, y, np.zeros(t.shape)), axis=1)


def parametric_circle_1(t):
    x = np.cos(t)
    y = np.sin(t)

    return np.stack((x, y, np.zeros(t.shape)), axis=1)


def render_urdf_template(template_path, models_dir, output_path=None):
    """
    Render a URDF Jinja2 template with absolute paths.

    Args:
        template_path (str or Path): Path to the Jinja2 template file (.j2)
        models_dir (str or Path): Absolute path to the directory containing STL model files
        output_path (str or Path, optional): Path where to save the rendered URDF.
                                           If None, returns the rendered content as string.

    Returns:
        str: Rendered URDF content if output_path is None, otherwise saves to file and returns the path
    """
    template_path = Path(template_path)
    models_dir = Path(models_dir).resolve()

    # Ensure models directory exists
    if not models_dir.exists():
        raise FileNotFoundError(f"Models directory does not exist: {models_dir}")

    # Ensure template file exists
    if not template_path.exists():
        raise FileNotFoundError(f"Template file does not exist: {template_path}")

    # Set up Jinja2 environment
    template_dir = template_path.parent
    template_name = template_path.name

    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template(template_name)

    # Render the template with absolute path
    rendered_content = template.render(models_dir=str(models_dir))

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            f.write(rendered_content)

        return str(output_path)
    else:
        return rendered_content


def render_robot_from_path(robot_path, output_path=None):
    """
    Load and render a robot URDF from a path containing a template and assets folder.

    Args:
        robot_path (str or Path): Path to directory containing robot.urdf.j2 template and assets folder
        output_path (str or Path, optional): Path where to save the rendered URDF.
                                           If None, returns the rendered content as string.

    Returns:
        str: Rendered URDF content if output_path is None, otherwise saves to file and returns the path

    Example:
        # Render empiric robot
        urdf_content = render_robot_from_path("robots/empiric")

        # Render vanilla robot and save to file
        render_robot_from_path("robots/vanilla", output_path="output.urdf")
    """
    robot_path = Path(robot_path)

    if not robot_path.exists():
        raise FileNotFoundError(f"Robot path does not exist: {robot_path}")

    # Find the template file (should be *.urdf.j2)
    template_files = list(robot_path.glob("*.urdf.j2"))
    if not template_files:
        raise FileNotFoundError(f"No .urdf.j2 template file found in: {robot_path}")
    if len(template_files) > 1:
        raise ValueError(f"Multiple .urdf.j2 template files found in: {robot_path}")

    template_path = template_files[0]

    assets_dir = robot_path / "assets"
    if not assets_dir.exists():
        raise FileNotFoundError(f"Assets directory not found: {assets_dir}")

    # Render the template
    return render_urdf_template(
        template_path=template_path, models_dir=assets_dir, output_path=output_path
    )


from contextlib import contextmanager
import tempfile


@contextmanager
def render_robot_from_template(robot_path):
    """
    Context manager that renders a robot template to a temporary file and yields the file path.

    Args:
        robot_path (str or Path): Path to directory containing robot.urdf.j2 template and assets folder

    Yields:
        Path: Path to temporary file containing the rendered URDF

    Example:
        with render_robot_from_template("robots/empiric") as robot_urdf:
            robot = VisualRobot(robot_urdf)
            robot.plot()
            robot.q = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
            # Temporary file is automatically cleaned up when exiting
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".urdf", delete=True
    ) as temp_file:
        # Render the robot URDF to the temporary file
        render_robot_from_path(robot_path, output_path=temp_file.name)

        # Yield the path to the temporary file as a Path object
        yield Path(temp_file.name)


def quaternion_to_rotation_matrix(quat):
    """
    Convert a Genesis quaternion (qw, qx, qy, qz) to a 3x3 rotation matrix.
    Handles PyTorch tensors (including CUDA) and numpy arrays/lists.
    """
    # Handle PyTorch tensors (including CUDA) without explicit import
    if hasattr(quat, "is_cuda") and quat.is_cuda:
        quat = quat.cpu()

    qw, qx, qy, qz = quat
    quat_xyzw = [qx, qy, qz, qw]
    rotation = R.from_quat(quat_xyzw)
    return rotation.as_matrix()


def print_collapsible(obj, max_length=500, title="Output", background_color="#f6f8fa"):
    """
    Display an object with smart handling of long outputs for GitHub-friendly viewing.

    If the string representation is longer than max_length, it wraps the output
    in an HTML collapsible section that works well on GitHub.

    Args:
        obj: Object to display
        max_length (int): Maximum length before wrapping in collapsible section
        title (str): Title for the collapsible section
        background_color (str): CSS color for the output background

    Example:
        # In a Jupyter notebook
        from manipylator.utils import print_collapsible

        # This will automatically wrap long outputs
        print_collapsible(your_long_symbolic_expression)

        # Customize the appearance
        print_collapsible(long_output, max_length=300, title="Detailed Results")
    """
    try:
        from IPython.display import display, HTML
    except ImportError:
        # Fallback if not in IPython environment
        print(obj)
        return

    obj_str = str(obj)

    if len(obj_str) > max_length:
        # Create collapsible HTML section for long outputs
        html_output = f"""
        <details style="margin: 10px 0; border: 1px solid #e1e4e8; border-radius: 6px;">
            <summary style="padding: 10px; background-color: {background_color}; cursor: pointer; font-weight: bold; border-bottom: 1px solid #e1e4e8;">
                {title} (click to expand - {len(obj_str)} characters)
            </summary>
            <pre style="margin: 0; padding: 15px; background-color: {background_color}; overflow-x: auto; font-size: 12px; line-height: 1.4; border-top: 1px solid #e1e4e8;">
            {obj_str}
            </pre>
        </details>
        """
        display(HTML(html_output))
    else:
        # For short outputs, display normally
        display(obj)
