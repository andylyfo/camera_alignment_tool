"""Railway camera alignment tool with interactive rail annotation."""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass, field

ESCAPE_KEY = 27
EPSILON = 1e-10
PAN_DIFF_THRESH = 0.05  # 5% difference
TILT_DIFF_THRESH = 0.02  # 2% difference
ROTATION_DIFF_THRESH = 1.5  # degrees
DARK_GREY_BGR = (40, 40, 40)
BRIGHT_GREEN_BGR = (0, 255, 0)
YELLOW_BGR = (0, 255, 255)
RED_BGR = (0, 0, 255)
BLUE_BGR = (255, 0, 0)
WHITE_BGR = (255, 255, 255)
CYAN_BGR = (0, 255, 255)
LIGHT_BLUE_BGR = (100, 200, 255)
LIGHT_GREY_BGR = (200, 200, 200)
MAGENTA_BGR = (255, 0, 255)
ORANGE_BGR = (0, 165, 255)
LINE_THICKNESS = 2
POINT_RADIUS = 5
VANISHING_POINT_RADIUS = 8
HORIZON_LINE_THICKNESS = 2
POLYGON_THICKNESS = 2
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
HEADER_FONT_SCALE = 0.7
METADATA_FONT_SCALE = 0.5
SUGGESTION_FONT_SCALE = 0.6
INSTRUCTION_FONT_SCALE = 0.5
HEADER_FONT_THICKNESS = 2
TEXT_FONT_THICKNESS = 1
METADATA_HEIGHT = 250
METADATA_Y_START = 30
LINE_HEIGHT = 25
COL1_X = 15
SUGGESTIONS_Y_START = 135
INSTRUCTION_Y_OFFSET = 10


@dataclass
class ImageState:
    """State for an image being annotated."""

    img: np.ndarray
    points: List[Tuple[int, int]] = field(default_factory=list)
    lines: List[Tuple[int, int, int, int]] = field(default_factory=list)
    vanishing_point: Optional[Tuple[float, float]] = None


class RailwayAlignmentTool:
    """Interactive tool for annotating rails and aligning cameras."""

    def __init__(self, img1_path: str, img2_path: str) -> None:
        """Initialize with two image paths."""
        img1_orig = cv2.imread(img1_path)
        img2_orig = cv2.imread(img2_path)

        if img1_orig is None or img2_orig is None:
            raise FileNotFoundError("Could not load one or both images")

        # resize images to match dimensions
        h1, w1 = img1_orig.shape[:2]
        h2, w2 = img2_orig.shape[:2]

        # use smaller img to avoid upscaling
        target_h = min(h1, h2)
        target_w = min(w1, w2)

        self.img1 = cv2.resize(img1_orig, (target_w, target_h))
        self.img2 = cv2.resize(img2_orig, (target_w, target_h))

        self.state1 = ImageState(img=self.img1.copy())
        self.state2 = ImageState(img=self.img2.copy())
        self.current_state = self.state1
        self.current_img_idx = 0

        self.window_name = "Camera Alignment Tool"
        self.polygon_mode = False  # Toggle between line mode and polygon mode

    def mouse_callback(self, event: int, x: int, y: int, flags: int, param: None) -> None:
        """Handle mouse events for drawing rail lines."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # determine which image was clicked
            h, w = self.state1.img.shape[:2]

            if x < w:
                # clicked on left image
                state = self.state1
                self.current_state = self.state1
                self.current_img_idx = 0
                click_x = x
            else:
                # clicked on right image
                state = self.state2
                self.current_state = self.state2
                self.current_img_idx = 1
                click_x = x - w

            state.points.append((click_x, y))

            # complete a line when we have 2 points
            if len(state.points) == 2:
                p1, p2 = state.points
                state.lines.append((p1[0], p1[1], p2[0], p2[1]))
                state.points.clear()

                if len(state.lines) == 2:
                    state.vanishing_point = self.find_intersection(state.lines[0], state.lines[1])

            self.update_display()

    @staticmethod
    def find_intersection(line1: Tuple[int, int, int, int], line2: Tuple[int, int, int, int]) -> Optional[Tuple[float, float]]:
        """Find intersection point of two lines."""
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

        if abs(denom) < EPSILON:
            return None

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom

        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)

        return x, y

    def calculate_vanishing_point(self) -> None:
        """Calculate vanishing point from two rail lines."""
        if len(self.current_state.lines) == 2:
            vp = self.find_intersection(self.current_state.lines[0], self.current_state.lines[1])
            self.current_state.vanishing_point = vp

    def calculate_line_angle(self, line: Tuple[int, int, int, int]) -> float:
        """Calculate angle of a line in degrees from horizontal (normalised bottom point to top)."""
        x1, y1, x2, y2 = line

        if y1 > y2:
            # p1 is bottom (near), p2 is top (far)
            angle_rad = np.arctan2(y2 - y1, x2 - x1)
        else:
            # p2 is bottom (near), p1 is top (far)
            angle_rad = np.arctan2(y1 - y2, x1 - x2)

        return np.degrees(angle_rad)

    @staticmethod
    def calculate_bottom_intersections(state: ImageState) -> Optional[Tuple[float, float]]:
        """Calculate where rails intersect the bottom edge of frame."""
        h, w = state.img.shape[:2]
        if len(state.lines) < 2:
            return None

        intersections = []
        for x1, y1, x2, y2 in state.lines:
            if abs(y2 - y1) < EPSILON:
                continue

            if x2 - x1 == 0:
                x_at_bottom = x2
            else:
                m = (y2 - y1) / (x2 - x1)
                c = y1 - m * x1
                x_at_bottom = (h - c) / m
            intersections.append(x_at_bottom)

        if len(intersections) == 2:
            return tuple(sorted(intersections))  # type: ignore
        return None

    def generate_alignment_suggestions(self) -> List[str]:
        """Generate camera alignment suggestions based on rail analysis."""
        suggestions = []

        if self.state1.vanishing_point is None or self.state2.vanishing_point is None or len(self.state1.lines) < 2 or len(self.state2.lines) < 2:
            return suggestions

        h, w = self.state1.img.shape[:2]

        # horizon difference analysis (tilt)
        vp_y1 = self.state1.vanishing_point[1]
        vp_y2 = self.state2.vanishing_point[1]
        horizon_diff = vp_y2 - vp_y1

        if abs(horizon_diff) > h * TILT_DIFF_THRESH:
            if horizon_diff > 0:
                suggestions.append(f"Cam2: TILT DOWN {abs(horizon_diff):.1f}px " "(horizon too high)")
            else:
                suggestions.append(f"Cam2: TILT UP {abs(horizon_diff):.1f}px " "(horizon too low)")

        # horizontal vanishing point difference (pan)
        vp_x1 = self.state1.vanishing_point[0]
        vp_x2 = self.state2.vanishing_point[0]
        horiz_diff = vp_x2 - vp_x1

        if abs(horiz_diff) > w * PAN_DIFF_THRESH:
            if horiz_diff > 0:
                suggestions.append(f"Cam2: PAN RIGHT {abs(horiz_diff):.1f}px " "(vanishing point too far right)")
            else:
                suggestions.append(f"Cam2: PAN LEFT {abs(horiz_diff):.1f}px " "(vanishing point too far left)")

        # get sorted rails (left to right based on average x position)
        left_rail1, right_rail1 = self.get_sorted_rails(self.state1.lines)
        left_rail2, right_rail2 = self.get_sorted_rails(self.state2.lines)

        # rail angle comparison
        left_angle1 = self.calculate_line_angle(left_rail1)
        right_angle1 = self.calculate_line_angle(right_rail1)
        left_angle2 = self.calculate_line_angle(left_rail2)
        right_angle2 = self.calculate_line_angle(right_rail2)

        left_angle_diff = left_angle2 - left_angle1
        right_angle_diff = right_angle2 - right_angle1

        if abs(left_angle_diff) > ROTATION_DIFF_THRESH or abs(right_angle_diff) > ROTATION_DIFF_THRESH:
            avg_angle_diff = (left_angle_diff + right_angle_diff) / 2
            if avg_angle_diff > 0:
                suggestions.append(f"Cam2: ROTATE CCW {abs(avg_angle_diff):.1f}deg " "(rails tilted clockwise)")
            else:
                suggestions.append(f"Cam2: ROTATE CW {abs(avg_angle_diff):.1f}deg " "(rails tilted counter-clockwise)")

        # rail gauge comparison (spacing at bottom of frame)
        bottom_intersections1 = self.calculate_bottom_intersections(self.state1)
        bottom_intersections2 = self.calculate_bottom_intersections(self.state2)

        if bottom_intersections1 is not None and bottom_intersections2 is not None:
            gauge1 = abs(bottom_intersections1[1] - bottom_intersections1[0])
            gauge2 = abs(bottom_intersections2[1] - bottom_intersections2[0])
            gauge_diff_pct = abs((gauge2 - gauge1) / gauge1) * 100

            sign = "-" if gauge1 > gauge2 else "+"

            # if gauge_diff_pct > 10:
            #     suggestions.append(
            #         f"Gauge mismatch: {sign}{gauge_diff_pct:.1f}% "
            #         f"(Cam1: {gauge1:.0f}px, Cam2: {gauge2:.0f}px)"
            #     )

        if not suggestions:
            suggestions.append("Cameras are well aligned!")

        return suggestions

    @staticmethod
    def get_sorted_rails(lines: List[Tuple[int, int, int, int]]) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]:
        """Sort rails left to right based on average x position."""
        if len(lines) != 2:
            return lines[0], lines[1]

        line1, line2 = lines
        avg_x1 = (line1[0] + line1[2]) / 2
        avg_x2 = (line2[0] + line2[2]) / 2

        if avg_x1 < avg_x2:
            return line1, line2
        else:
            return line2, line1

    @staticmethod
    def extract_polygon_points(state: ImageState) -> Optional[np.ndarray]:
        """Extract four corner points from two lines to form a polygon."""
        if len(state.lines) != 2:
            return None

        # Get all four points from the two lines
        line1 = state.lines[0]
        line2 = state.lines[1]

        points = np.array([
            [line1[0], line1[1]],  # line1 point1
            [line1[2], line1[3]],  # line1 point2
            [line2[0], line2[1]],  # line2 point1
            [line2[2], line2[3]]   # line2 point2
        ], dtype=np.int32)

        return points

    @staticmethod
    def draw_polygon_overlay(img: np.ndarray, own_points: Optional[np.ndarray],
                            other_points: Optional[np.ndarray], is_own: bool = True) -> np.ndarray:
        """Draw polygon overlay on image."""
        result = img.copy()

        # Draw own polygon (solid)
        if own_points is not None:
            own_color = BRIGHT_GREEN_BGR if is_own else LIGHT_BLUE_BGR
            cv2.polylines(result, [own_points], isClosed=True,
                         color=own_color, thickness=POLYGON_THICKNESS)
            # Draw filled semi-transparent polygon
            overlay = result.copy()
            cv2.fillPoly(overlay, [own_points], own_color)
            cv2.addWeighted(overlay, 0.15, result, 0.85, 0, result)

            # Draw points
            for point in own_points:
                cv2.circle(result, tuple(point), POINT_RADIUS, own_color, -1)

        # Draw other image's polygon overlay (dashed/dotted appearance)
        if other_points is not None:
            other_color = MAGENTA_BGR
            # Draw dashed lines by drawing segments
            for i in range(len(other_points)):
                pt1 = tuple(other_points[i])
                pt2 = tuple(other_points[(i + 1) % len(other_points)])

                # Calculate line length and draw dashed line
                length = np.linalg.norm(np.array(pt1) - np.array(pt2))
                dash_length = 10
                gap_length = 5
                num_dashes = int(length / (dash_length + gap_length))

                for j in range(num_dashes):
                    start_ratio = j * (dash_length + gap_length) / length
                    end_ratio = (j * (dash_length + gap_length) + dash_length) / length

                    start_pt = (
                        int(pt1[0] + (pt2[0] - pt1[0]) * start_ratio),
                        int(pt1[1] + (pt2[1] - pt1[1]) * start_ratio)
                    )
                    end_pt = (
                        int(pt1[0] + (pt2[0] - pt1[0]) * end_ratio),
                        int(pt1[1] + (pt2[1] - pt1[1]) * end_ratio)
                    )

                    cv2.line(result, start_pt, end_pt, other_color, POLYGON_THICKNESS)

            # Draw points as hollow circles
            for point in other_points:
                cv2.circle(result, tuple(point), POINT_RADIUS, other_color, 2)

        return result

    @staticmethod
    def draw_annotations(img: np.ndarray, state: ImageState) -> np.ndarray:
        """Draw all annotations on an image."""
        result = img.copy()

        # draw completed lines
        for x1, y1, x2, y2 in state.lines:
            cv2.line(result, (x1, y1), (x2, y2), BRIGHT_GREEN_BGR, LINE_THICKNESS)

        # draw current incomplete line
        if len(state.points) == 1:
            cv2.circle(result, state.points[0], POINT_RADIUS, YELLOW_BGR, -1)

        # draw vanishing point
        if state.vanishing_point is not None:
            vp_x, vp_y = state.vanishing_point
            cv2.circle(result, (int(vp_x), int(vp_y)), VANISHING_POINT_RADIUS, RED_BGR, -1)

        return result

    def update_display(self) -> None:
        """Update the display with current annotations."""
        if self.polygon_mode:
            # Extract polygons from both images
            poly1 = self.extract_polygon_points(self.state1)
            poly2 = self.extract_polygon_points(self.state2)

            # Draw polygons with overlays
            annotated1 = self.draw_polygon_overlay(self.state1.img, poly1, poly2, is_own=True)
            annotated2 = self.draw_polygon_overlay(self.state2.img, poly2, poly1, is_own=True)

            # Draw current incomplete points in polygon mode
            if len(self.state1.points) == 1:
                cv2.circle(annotated1, self.state1.points[0], POINT_RADIUS, YELLOW_BGR, -1)
            if len(self.state2.points) == 1:
                cv2.circle(annotated2, self.state2.points[0], POINT_RADIUS, YELLOW_BGR, -1)
        else:
            # Original line drawing mode
            annotated1 = self.draw_annotations(self.state1.img, self.state1)
            annotated2 = self.draw_annotations(self.state2.img, self.state2)

        h, w = annotated1.shape[:2]

        combined = np.hstack([annotated1, annotated2])

        # Only draw horizon lines in line mode
        if not self.polygon_mode:
            if self.state1.vanishing_point is not None:
                vp_y1 = int(self.state1.vanishing_point[1])
                cv2.line(combined, (0, vp_y1), (w, vp_y1), BLUE_BGR, HORIZON_LINE_THICKNESS)

            if self.state2.vanishing_point is not None:
                vp_y2 = int(self.state2.vanishing_point[1])
                cv2.line(combined, (w, vp_y2), (w * 2, vp_y2), BLUE_BGR, HORIZON_LINE_THICKNESS)

        # metadata panel
        metadata_panel = np.zeros((METADATA_HEIGHT, w * 2, 3), dtype=np.uint8)
        metadata_panel[:] = DARK_GREY_BGR

        y_pos = METADATA_Y_START
        col2_x = w + COL1_X

        # columns
        col1_colour = BRIGHT_GREEN_BGR
        col2_colour = LIGHT_BLUE_BGR

        if self.current_img_idx == 1:
            col1_colour = LIGHT_BLUE_BGR
            col2_colour = BRIGHT_GREEN_BGR

        cv2.putText(metadata_panel, "Camera 1", (COL1_X, y_pos), TEXT_FONT, HEADER_FONT_SCALE, col1_colour, HEADER_FONT_THICKNESS)
        cv2.putText(metadata_panel, "Camera 2", (col2_x, y_pos), TEXT_FONT, HEADER_FONT_SCALE, col2_colour, HEADER_FONT_THICKNESS)
        y_pos += LINE_HEIGHT + 5

        # cam1 metadata
        if len(self.state1.lines) >= 2:
            left_rail, right_rail = self.get_sorted_rails(self.state1.lines)
            left_angle = self.calculate_line_angle(left_rail)
            right_angle = self.calculate_line_angle(right_rail)

            cv2.putText(metadata_panel, f"Left Rail: {left_angle:.1f}deg", (COL1_X, y_pos), TEXT_FONT, METADATA_FONT_SCALE, WHITE_BGR, TEXT_FONT_THICKNESS)
            y_pos += LINE_HEIGHT
            cv2.putText(metadata_panel, f"Right Rail: {right_angle:.1f}deg", (COL1_X, y_pos), TEXT_FONT, METADATA_FONT_SCALE, WHITE_BGR, TEXT_FONT_THICKNESS)
            y_pos += LINE_HEIGHT
        elif len(self.state1.lines) >= 1:
            y_pos += LINE_HEIGHT * 2
        else:
            y_pos += LINE_HEIGHT * 2

        if self.state1.vanishing_point is not None:
            vp_y1_pct = (self.state1.vanishing_point[1] / h) * 100
            cv2.putText(metadata_panel, f"Horizon: {vp_y1_pct:.1f}%", (COL1_X, y_pos), TEXT_FONT, METADATA_FONT_SCALE, WHITE_BGR, TEXT_FONT_THICKNESS)

        # cam2 metadata
        y_pos = METADATA_Y_START + LINE_HEIGHT + 5
        if len(self.state2.lines) >= 2:
            left_rail, right_rail = self.get_sorted_rails(self.state2.lines)
            left_angle = self.calculate_line_angle(left_rail)
            right_angle = self.calculate_line_angle(right_rail)

            cv2.putText(metadata_panel, f"Left Rail: {left_angle:.1f}deg", (col2_x, y_pos), TEXT_FONT, METADATA_FONT_SCALE, WHITE_BGR, TEXT_FONT_THICKNESS)
            y_pos += LINE_HEIGHT
            cv2.putText(metadata_panel, f"Right Rail: {right_angle:.1f}deg", (col2_x, y_pos), TEXT_FONT, METADATA_FONT_SCALE, WHITE_BGR, TEXT_FONT_THICKNESS)
            y_pos += LINE_HEIGHT
        elif len(self.state2.lines) >= 1:
            y_pos += LINE_HEIGHT * 2
        else:
            y_pos += LINE_HEIGHT * 2

        if self.state2.vanishing_point is not None:
            vp_y2_pct = (self.state2.vanishing_point[1] / h) * 100
            cv2.putText(metadata_panel, f"Horizon: {vp_y2_pct:.1f}%", (col2_x, y_pos), TEXT_FONT, METADATA_FONT_SCALE, WHITE_BGR, TEXT_FONT_THICKNESS)

        # suggestions
        suggestions = self.generate_alignment_suggestions()
        if suggestions:
            y_pos = SUGGESTIONS_Y_START
            cv2.putText(metadata_panel, "Alignment Suggestions:", (COL1_X, y_pos), TEXT_FONT, SUGGESTION_FONT_SCALE, CYAN_BGR, HEADER_FONT_THICKNESS)
            y_pos += LINE_HEIGHT

            for suggestion in suggestions:
                color = BRIGHT_GREEN_BGR if "well aligned" in suggestion else LIGHT_BLUE_BGR
                cv2.putText(metadata_panel, f"  {suggestion}", (COL1_X, y_pos), TEXT_FONT, METADATA_FONT_SCALE, color, TEXT_FONT_THICKNESS)
                y_pos += LINE_HEIGHT - 5

        display = np.vstack([combined, metadata_panel])

        # Update instructions based on mode
        mode_text = "[POLYGON MODE]" if self.polygon_mode else "[LINE MODE]"
        instruction_text = f"{mode_text} Click: draw | P: toggle mode | U: undo | R: reset | Q: quit"

        cv2.putText(
            display,
            instruction_text,
            (INSTRUCTION_Y_OFFSET, h + METADATA_HEIGHT - INSTRUCTION_Y_OFFSET),
            TEXT_FONT,
            INSTRUCTION_FONT_SCALE,
            LIGHT_GREY_BGR,
            TEXT_FONT_THICKNESS,
        )

        cv2.imshow(self.window_name, display)

    def toggle_polygon_mode(self) -> None:
        """Toggle between line mode and polygon mode."""
        self.polygon_mode = not self.polygon_mode
        self.update_display()

    def undo_last_action(self) -> None:
        """Undo the last action (point or line)."""
        # TODO this doesn't quite work - use reset for now
        if self.current_state.points:
            self.current_state.points.pop()
        elif self.current_state.lines:
            self.current_state.lines.pop()
            self.current_state.vanishing_point = None
            if len(self.current_state.lines) == 2:
                self.current_state.vanishing_point = self.find_intersection(self.current_state.lines[0], self.current_state.lines[1])
        self.update_display()

    def reset_current_image(self) -> None:
        """Reset annotations for current image."""
        self.current_state.points.clear()
        self.current_state.lines.clear()
        self.current_state.vanishing_point = None
        self.update_display()

    def reset_all(self) -> None:
        """Reset annotations for both images."""
        self.state1.points.clear()
        self.state1.lines.clear()
        self.state1.vanishing_point = None
        self.state2.points.clear()
        self.state2.lines.clear()
        self.state2.vanishing_point = None
        self.update_display()

    def switch_image(self) -> None:
        """Switch between annotating image 1 and image 2."""
        self.current_img_idx = 1 - self.current_img_idx
        self.current_state = self.state1 if self.current_img_idx == 0 else self.state2
        self.update_display()

    def run(self) -> None:
        """Run the interactive annotation tool."""
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        self.update_display()

        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == ESCAPE_KEY:
                break
            elif key == ord("u"):
                self.undo_last_action()
            elif key == ord("r"):
                self.reset_all()
            elif key == ord("p"):
                self.toggle_polygon_mode()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys

    img1 = sys.argv[1] if len(sys.argv) > 1 else "camera1.png"
    img2 = sys.argv[2] if len(sys.argv) > 2 else "camera2.png"

    tool = RailwayAlignmentTool(img1, img2)
    tool.run()
