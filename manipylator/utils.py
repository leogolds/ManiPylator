import numpy as np
import os
from pathlib import Path
from jinja2 import Environment, FileSystemLoader


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
        
        with open(output_path, 'w') as f:
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
        template_path=template_path,
        models_dir=assets_dir,
        output_path=output_path
    )
