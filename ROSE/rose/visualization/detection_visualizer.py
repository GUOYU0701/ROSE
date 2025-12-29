"""
3D Detection visualization utilities
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.patches as patches
from collections import defaultdict
import os
from pathlib import Path


class DetectionVisualizer:
    """
    Visualization utilities for 3D object detection results
    """
    
    def __init__(self):
        # Color map for different classes
        self.class_colors = {
            'Car': (0, 255, 0),        # Green
            'Pedestrian': (255, 0, 0),  # Red  
            'Cyclist': (0, 0, 255),     # Blue
            'Motorcyclist': (255, 255, 0),  # Yellow
            'Trafficcone': (255, 0, 255)    # Magenta
        }
        
        # Default colors for prediction vs ground truth
        self.pred_color = (0, 255, 0)     # Green for predictions
        self.gt_color = (255, 0, 0)       # Red for ground truth
        
        # Enhanced visualization for problematic classes
        self.problematic_classes = ['Pedestrian', 'Cyclist']
        self.enhanced_colors = {
            'Pedestrian': (255, 100, 100),  # Light red for better visibility
            'Cyclist': (100, 100, 255),     # Light blue for better visibility
        }
        
        # Performance tracking
        self.detection_stats = {
            'total_detections': 0,
            'class_detections': defaultdict(int),
            'confidence_distributions': defaultdict(list),
            'size_distributions': defaultdict(list),
            'distance_distributions': defaultdict(list)
        }
    
    def draw_3d_boxes_on_image(self,
                              image: np.ndarray,
                              predictions: Dict[str, Any],
                              ground_truth: Optional[Dict[str, Any]] = None,
                              class_names: Optional[List[str]] = None,
                              confidence_threshold: float = 0.3) -> np.ndarray:
        """
        Draw 3D bounding boxes projected onto image
        
        Args:
            image: Input image (H, W, 3)
            predictions: Model predictions with bboxes_3d, scores_3d, labels_3d
            ground_truth: Optional ground truth annotations
            class_names: List of class names
            confidence_threshold: Minimum confidence for visualization
            
        Returns:
            Image with 3D boxes drawn
        """
        img_vis = image.copy()
        
        class_names = class_names or ['Pedestrian', 'Cyclist', 'Car']
        
        # Draw predictions
        if 'bboxes_3d' in predictions and 'scores_3d' in predictions:
            bboxes_3d = predictions['bboxes_3d']
            scores = predictions['scores_3d'] 
            labels = predictions.get('labels_3d', [])
            
            # Filter by confidence
            valid_indices = scores > confidence_threshold
            
            if hasattr(valid_indices, 'numpy'):
                valid_indices = valid_indices.numpy()
            if hasattr(bboxes_3d, 'numpy'):
                bboxes_3d = bboxes_3d.numpy()
            if hasattr(scores, 'numpy'):
                scores = scores.numpy()
            if hasattr(labels, 'numpy'):
                labels = labels.numpy()
            
            for i, (bbox, score, label) in enumerate(zip(bboxes_3d, scores, labels)):
                if not valid_indices[i]:
                    continue
                    
                class_name = class_names[label] if label < len(class_names) else f'Class_{label}'
                color = self.class_colors.get(class_name, self.pred_color)
                
                # Project 3D box to image (simplified projection)
                img_vis = self._draw_3d_box_on_image(
                    img_vis, bbox, color, 
                    f'{class_name}: {score:.2f}', is_prediction=True
                )
        
        # Draw ground truth if available
        if ground_truth and 'bboxes_3d' in ground_truth:
            gt_bboxes_3d = ground_truth['bboxes_3d']
            gt_labels = ground_truth.get('labels_3d', [])
            
            if hasattr(gt_bboxes_3d, 'numpy'):
                gt_bboxes_3d = gt_bboxes_3d.numpy()
            if hasattr(gt_labels, 'numpy'):
                gt_labels = gt_labels.numpy()
            
            for bbox, label in zip(gt_bboxes_3d, gt_labels):
                class_name = class_names[label] if label < len(class_names) else f'Class_{label}'
                
                img_vis = self._draw_3d_box_on_image(
                    img_vis, bbox, self.gt_color,
                    f'GT: {class_name}', is_prediction=False
                )
        
        return img_vis
    
    def _draw_3d_box_on_image(self,
                            image: np.ndarray,
                            bbox_3d: np.ndarray,
                            color: Tuple[int, int, int],
                            label: str,
                            is_prediction: bool = True) -> np.ndarray:
        """
        Draw a single 3D bounding box on image
        
        Args:
            image: Input image
            bbox_3d: 3D bounding box (x, y, z, w, h, l, r)
            color: Color for the box
            label: Text label
            is_prediction: Whether this is prediction (solid) or GT (dashed)
            
        Returns:
            Image with box drawn
        """
        if len(bbox_3d) < 7:
            return image
        
        x, y, z, w, h, l, r = bbox_3d[:7]
        
        # Simple projection: just draw 2D bounding box at object center
        # In practice, you would use camera calibration for proper 3D->2D projection
        
        # Project 3D center to image coordinates (simplified)
        # This assumes a simple projection model - replace with actual calibration
        img_h, img_w = image.shape[:2]
        
        # Simple mapping from world coordinates to image coordinates
        # This is a placeholder - real implementation needs camera matrix
        img_x = int((x / 70.4) * img_w / 2 + img_w / 2)  # Normalize by max range
        img_y = int((1 - (y + 40) / 80) * img_h)  # Normalize by range
        
        # Draw 2D bounding box approximation
        box_width = int(w * 20)  # Scale factor for visualization
        box_height = int(h * 20)
        
        x1 = max(0, img_x - box_width // 2)
        y1 = max(0, img_y - box_height // 2)
        x2 = min(img_w - 1, img_x + box_width // 2)
        y2 = min(img_h - 1, img_y + box_height // 2)
        
        # Draw rectangle
        line_thickness = 2 if is_prediction else 3
        cv2.rectangle(image, (x1, y1), (x2, y2), color, line_thickness)
        
        # Add label
        font_scale = 0.5
        font_thickness = 1
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        
        # Background rectangle for text
        cv2.rectangle(image, (x1, y1 - text_height - 5), (x1 + text_width, y1), color, -1)
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                   font_scale, (255, 255, 255), font_thickness)
        
        return image
    
    def visualize_3d_boxes_on_pointcloud(self,
                                       points: np.ndarray,
                                       predictions: Dict[str, Any],
                                       ground_truth: Optional[Dict[str, Any]] = None,
                                       output_path: Optional[str] = None,
                                       class_names: Optional[List[str]] = None,
                                       confidence_threshold: float = 0.3,
                                       point_size: float = 0.5) -> None:
        """
        Visualize 3D bounding boxes on point cloud
        
        Args:
            points: Point cloud data (N, 3 or 4)
            predictions: Model predictions
            ground_truth: Optional ground truth
            output_path: Path to save visualization
            class_names: List of class names
            confidence_threshold: Minimum confidence
            point_size: Size of points in visualization
        """
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        class_names = class_names or ['Pedestrian', 'Cyclist', 'Car']
        
        # Plot point cloud
        if points.shape[1] >= 3:
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                      c='gray', s=point_size, alpha=0.3)
        
        # Plot prediction boxes
        if 'bboxes_3d' in predictions and 'scores_3d' in predictions:
            bboxes_3d = predictions['bboxes_3d']
            scores = predictions['scores_3d']
            labels = predictions.get('labels_3d', [])
            
            # Convert to numpy if needed
            if hasattr(bboxes_3d, 'numpy'):
                bboxes_3d = bboxes_3d.numpy()
            if hasattr(scores, 'numpy'):
                scores = scores.numpy()
            if hasattr(labels, 'numpy'):
                labels = labels.numpy()
            
            # Filter by confidence
            valid_indices = scores > confidence_threshold
            
            for i, (bbox, score, label) in enumerate(zip(bboxes_3d, scores, labels)):
                if not valid_indices[i]:
                    continue
                
                class_name = class_names[label] if label < len(class_names) else f'Class_{label}'
                color = self._get_class_color_rgb(class_name)
                
                self._plot_3d_box(ax, bbox, color, f'{class_name}: {score:.2f}', 
                                linestyle='-', linewidth=2)
        
        # Plot ground truth boxes
        if ground_truth and 'bboxes_3d' in ground_truth:
            gt_bboxes_3d = ground_truth['bboxes_3d']
            gt_labels = ground_truth.get('labels_3d', [])
            
            if hasattr(gt_bboxes_3d, 'numpy'):
                gt_bboxes_3d = gt_bboxes_3d.numpy()
            if hasattr(gt_labels, 'numpy'):
                gt_labels = gt_labels.numpy()
            
            for bbox, label in zip(gt_bboxes_3d, gt_labels):
                class_name = class_names[label] if label < len(class_names) else f'Class_{label}'
                
                self._plot_3d_box(ax, bbox, 'red', f'GT: {class_name}',
                                linestyle='--', linewidth=3)
        
        # Set labels and title
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('3D Detection Results')
        
        # Set equal aspect ratio
        max_range = 50
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-3, 5])
        
        # Add legend
        ax.legend()
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def _plot_3d_box(self,
                    ax,
                    bbox_3d: np.ndarray,
                    color: str,
                    label: str,
                    linestyle: str = '-',
                    linewidth: int = 2) -> None:
        """
        Plot a single 3D bounding box
        
        Args:
            ax: 3D matplotlib axis
            bbox_3d: 3D box parameters (x, y, z, w, h, l, r)
            color: Color for the box
            label: Label for the box
            linestyle: Line style
            linewidth: Line width
        """
        if len(bbox_3d) < 7:
            return
        
        x, y, z, w, h, l, r = bbox_3d[:7]
        
        # Generate box corners
        corners = self._generate_3d_box_corners(x, y, z, w, h, l, r)
        
        # Define the 12 edges of a 3D box
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
            [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
            [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
        ]
        
        # Plot edges
        for edge in edges:
            points = corners[edge]
            ax.plot3D(points[:, 0], points[:, 1], points[:, 2], 
                     color=color, linestyle=linestyle, linewidth=linewidth)
        
        # Add label at box center
        ax.text(x, y, z + h/2, label, color=color)
    
    def _generate_3d_box_corners(self,
                               x: float, y: float, z: float,
                               w: float, h: float, l: float,
                               r: float) -> np.ndarray:
        """
        Generate 8 corners of a 3D bounding box
        
        Args:
            x, y, z: Center coordinates
            w, h, l: Width, height, length
            r: Rotation angle around z-axis
            
        Returns:
            Array of 8 corner coordinates (8, 3)
        """
        # Create box corners in local coordinate system
        corners = np.array([
            [-w/2, -l/2, -h/2],
            [ w/2, -l/2, -h/2],
            [ w/2,  l/2, -h/2],
            [-w/2,  l/2, -h/2],
            [-w/2, -l/2,  h/2],
            [ w/2, -l/2,  h/2],
            [ w/2,  l/2,  h/2],
            [-w/2,  l/2,  h/2]
        ])
        
        # Apply rotation around z-axis
        cos_r, sin_r = np.cos(r), np.sin(r)
        rotation_matrix = np.array([
            [cos_r, -sin_r, 0],
            [sin_r,  cos_r, 0],
            [0,      0,     1]
        ])
        
        corners = corners @ rotation_matrix.T
        
        # Translate to global coordinates
        corners[:, 0] += x
        corners[:, 1] += y
        corners[:, 2] += z
        
        return corners
    
    def _get_class_color_rgb(self, class_name: str) -> str:
        """Get RGB color string for matplotlib"""
        bgr_color = self.class_colors.get(class_name, (0, 255, 0))
        # Convert BGR to RGB and normalize
        rgb_color = (bgr_color[2]/255, bgr_color[1]/255, bgr_color[0]/255)
        return rgb_color
    
    def create_detection_comparison(self,
                                  clean_results: Dict[str, Any],
                                  augmented_results: Dict[str, Any],
                                  image_clean: np.ndarray,
                                  image_augmented: np.ndarray,
                                  output_path: str,
                                  class_names: Optional[List[str]] = None) -> None:
        """
        Create side-by-side comparison of detection results on clean vs augmented data
        
        Args:
            clean_results: Detection results on clean data
            augmented_results: Detection results on augmented data
            image_clean: Clean input image
            image_augmented: Augmented input image
            output_path: Path to save comparison
            class_names: List of class names
        """
        # Draw boxes on both images
        img_clean_vis = self.draw_3d_boxes_on_image(
            image_clean, clean_results, class_names=class_names
        )
        img_aug_vis = self.draw_3d_boxes_on_image(
            image_augmented, augmented_results, class_names=class_names
        )
        
        # Create side-by-side comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        ax1.imshow(cv2.cvtColor(img_clean_vis, cv2.COLOR_BGR2RGB))
        ax1.set_title('Detection on Clean Data')
        ax1.axis('off')
        
        ax2.imshow(cv2.cvtColor(img_aug_vis, cv2.COLOR_BGR2RGB))
        ax2.set_title('Detection on Augmented Data')
        ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def analyze_detection_performance(self, predictions: Dict[str, Any],
                                    ground_truth: Dict[str, Any],
                                    class_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze detection performance with focus on problematic classes
        """
        class_names = class_names or ['Pedestrian', 'Cyclist', 'Car']
        analysis = {
            'overall_stats': {},
            'class_specific_stats': {},
            'problematic_class_analysis': {},
            'detection_quality_metrics': {}
        }
        
        if 'bboxes_3d' not in predictions or 'bboxes_3d' not in ground_truth:
            return analysis
        
        pred_bboxes = predictions['bboxes_3d']
        pred_scores = predictions.get('scores_3d', [])
        pred_labels = predictions.get('labels_3d', [])
        
        gt_bboxes = ground_truth['bboxes_3d']
        gt_labels = ground_truth.get('labels_3d', [])
        
        # Convert to numpy if needed
        for arr_name, arr in [('pred_bboxes', pred_bboxes), ('pred_scores', pred_scores), 
                             ('pred_labels', pred_labels), ('gt_bboxes', gt_bboxes), 
                             ('gt_labels', gt_labels)]:
            if hasattr(arr, 'numpy'):
                locals()[arr_name] = arr.numpy()
        
        # Overall statistics
        analysis['overall_stats'] = {
            'total_predictions': len(pred_bboxes) if len(pred_bboxes.shape) > 0 else 0,
            'total_ground_truth': len(gt_bboxes) if len(gt_bboxes.shape) > 0 else 0,
            'avg_confidence': np.mean(pred_scores) if len(pred_scores) > 0 else 0.0
        }
        
        # Class-specific analysis
        for class_idx, class_name in enumerate(class_names):
            class_analysis = self._analyze_class_performance(
                pred_bboxes, pred_scores, pred_labels,
                gt_bboxes, gt_labels, class_idx
            )
            analysis['class_specific_stats'][class_name] = class_analysis
            
            # Special analysis for problematic classes
            if class_name in self.problematic_classes:
                problematic_analysis = self._analyze_problematic_class(
                    pred_bboxes, pred_scores, pred_labels,
                    gt_bboxes, gt_labels, class_idx, class_name
                )
                analysis['problematic_class_analysis'][class_name] = problematic_analysis
        
        # Detection quality metrics
        analysis['detection_quality_metrics'] = self._compute_detection_quality_metrics(
            pred_bboxes, pred_scores, pred_labels, gt_bboxes, gt_labels, class_names
        )
        
        return analysis
    
    def _analyze_class_performance(self, pred_bboxes, pred_scores, pred_labels,
                                 gt_bboxes, gt_labels, class_idx) -> Dict[str, Any]:
        """Analyze performance for specific class"""
        # Filter predictions and ground truth for this class
        class_pred_mask = pred_labels == class_idx if len(pred_labels) > 0 else []
        class_gt_mask = gt_labels == class_idx if len(gt_labels) > 0 else []
        
        class_pred_count = np.sum(class_pred_mask) if len(class_pred_mask) > 0 else 0
        class_gt_count = np.sum(class_gt_mask) if len(class_gt_mask) > 0 else 0
        
        analysis = {
            'prediction_count': class_pred_count,
            'ground_truth_count': class_gt_count,
            'detection_rate': class_pred_count / max(class_gt_count, 1),
            'avg_confidence': 0.0,
            'confidence_distribution': {},
            'size_analysis': {},
            'distance_analysis': {}
        }
        
        if class_pred_count > 0:
            class_pred_scores = pred_scores[class_pred_mask]
            analysis['avg_confidence'] = np.mean(class_pred_scores)
            
            # Confidence distribution
            analysis['confidence_distribution'] = {
                'high_conf_count': np.sum(class_pred_scores > 0.7),
                'medium_conf_count': np.sum((class_pred_scores > 0.3) & (class_pred_scores <= 0.7)),
                'low_conf_count': np.sum(class_pred_scores <= 0.3)
            }
        
        if class_gt_count > 0:
            class_gt_bboxes = gt_bboxes[class_gt_mask]
            
            # Size analysis
            if len(class_gt_bboxes.shape) > 1 and class_gt_bboxes.shape[1] >= 6:
                volumes = class_gt_bboxes[:, 3] * class_gt_bboxes[:, 4] * class_gt_bboxes[:, 5]  # w*h*l
                analysis['size_analysis'] = {
                    'avg_volume': np.mean(volumes),
                    'min_volume': np.min(volumes),
                    'max_volume': np.max(volumes),
                    'small_object_count': np.sum(volumes < np.percentile(volumes, 33)),
                }
            
            # Distance analysis
            if len(class_gt_bboxes.shape) > 1 and class_gt_bboxes.shape[1] >= 3:
                distances = np.sqrt(class_gt_bboxes[:, 0]**2 + class_gt_bboxes[:, 1]**2)
                analysis['distance_analysis'] = {
                    'avg_distance': np.mean(distances),
                    'near_objects': np.sum(distances < 20),  # < 20m
                    'far_objects': np.sum(distances > 50),   # > 50m
                }
        
        return analysis
    
    def _analyze_problematic_class(self, pred_bboxes, pred_scores, pred_labels,
                                 gt_bboxes, gt_labels, class_idx, class_name) -> Dict[str, Any]:
        """Special analysis for problematic classes (Pedestrian, Cyclist)"""
        analysis = {
            'detection_challenges': [],
            'failure_patterns': [],
            'improvement_suggestions': [],
            'augmentation_sensitivity': {}
        }
        
        class_pred_mask = pred_labels == class_idx if len(pred_labels) > 0 else []
        class_gt_mask = gt_labels == class_idx if len(gt_labels) > 0 else []
        
        class_pred_count = np.sum(class_pred_mask) if len(class_pred_mask) > 0 else 0
        class_gt_count = np.sum(class_gt_mask) if len(class_gt_mask) > 0 else 0
        
        # Identify detection challenges
        if class_gt_count > 0:
            detection_rate = class_pred_count / class_gt_count
            
            if detection_rate < 0.3:
                analysis['detection_challenges'].append('Very low detection rate')
                analysis['improvement_suggestions'].append('Increase anchor density for small objects')
                analysis['improvement_suggestions'].append('Use multi-scale training')
            elif detection_rate < 0.5:
                analysis['detection_challenges'].append('Low detection rate')
                analysis['improvement_suggestions'].append('Adjust loss weights for this class')
            
            # Analyze object characteristics that lead to failures
            if len(gt_bboxes.shape) > 1 and class_gt_count > 0:
                class_gt_bboxes = gt_bboxes[class_gt_mask]
                
                # Small object analysis
                if class_gt_bboxes.shape[1] >= 6:
                    volumes = class_gt_bboxes[:, 3] * class_gt_bboxes[:, 4] * class_gt_bboxes[:, 5]
                    small_obj_ratio = np.sum(volumes < np.percentile(volumes, 50)) / len(volumes)
                    
                    if small_obj_ratio > 0.7:
                        analysis['failure_patterns'].append('High ratio of small objects')
                        analysis['improvement_suggestions'].append('Focus augmentation on small object regions')
                
                # Distance analysis
                if class_gt_bboxes.shape[1] >= 3:
                    distances = np.sqrt(class_gt_bboxes[:, 0]**2 + class_gt_bboxes[:, 1]**2)
                    far_obj_ratio = np.sum(distances > 40) / len(distances)
                    
                    if far_obj_ratio > 0.5:
                        analysis['failure_patterns'].append('Many objects at far distances')
                        analysis['improvement_suggestions'].append('Improve long-range point cloud processing')
        
        # Confidence analysis for predictions
        if class_pred_count > 0:
            class_pred_scores = pred_scores[class_pred_mask]
            low_conf_ratio = np.sum(class_pred_scores < 0.5) / len(class_pred_scores)
            
            if low_conf_ratio > 0.6:
                analysis['failure_patterns'].append('Low confidence predictions')
                analysis['improvement_suggestions'].append('Increase SSL training for this class')
        
        # Class-specific recommendations
        if class_name == 'Pedestrian':
            analysis['improvement_suggestions'].extend([
                'Use human pose-aware augmentation',
                'Increase cross-modal alignment for human shapes',
                'Apply class-specific NMS thresholds'
            ])
        elif class_name == 'Cyclist':
            analysis['improvement_suggestions'].extend([
                'Focus on bicycle+person combined features',
                'Use motion-aware augmentation for cyclists',
                'Improve handling of partially occluded cyclists'
            ])
        
        return analysis
    
    def _compute_detection_quality_metrics(self, pred_bboxes, pred_scores, pred_labels,
                                         gt_bboxes, gt_labels, class_names) -> Dict[str, Any]:
        """Compute overall detection quality metrics"""
        quality_metrics = {
            'prediction_quality': {},
            'recall_estimation': {},
            'precision_estimation': {},
            'confidence_calibration': {}
        }
        
        if len(pred_scores) > 0:
            # Confidence distribution analysis
            quality_metrics['confidence_calibration'] = {
                'high_confidence_ratio': np.sum(pred_scores > 0.8) / len(pred_scores),
                'medium_confidence_ratio': np.sum((pred_scores > 0.5) & (pred_scores <= 0.8)) / len(pred_scores),
                'low_confidence_ratio': np.sum(pred_scores <= 0.5) / len(pred_scores),
                'avg_confidence': np.mean(pred_scores),
                'confidence_std': np.std(pred_scores)
            }
            
            # Prediction quality by confidence threshold
            for threshold in [0.3, 0.5, 0.7]:
                high_conf_preds = np.sum(pred_scores > threshold)
                quality_metrics['prediction_quality'][f'predictions_above_{threshold}'] = high_conf_preds
        
        # Class-wise quality estimation
        for class_idx, class_name in enumerate(class_names):
            class_pred_mask = pred_labels == class_idx if len(pred_labels) > 0 else []
            class_gt_mask = gt_labels == class_idx if len(gt_labels) > 0 else []
            
            class_pred_count = np.sum(class_pred_mask) if len(class_pred_mask) > 0 else 0
            class_gt_count = np.sum(class_gt_mask) if len(class_gt_mask) > 0 else 0
            
            # Simple recall estimation (without IoU matching)
            recall_estimate = min(class_pred_count / max(class_gt_count, 1), 1.0)
            quality_metrics['recall_estimation'][class_name] = recall_estimate
            
            # Precision estimation (simplified)
            if class_pred_count > 0:
                precision_estimate = min(class_gt_count / class_pred_count, 1.0)
                quality_metrics['precision_estimation'][class_name] = precision_estimate
        
        return quality_metrics
    
    def create_performance_summary_plot(self, analysis_results: Dict[str, Any],
                                      output_path: str) -> None:
        """Create comprehensive performance summary visualization"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Detection Performance Analysis', fontsize=16)
        
        # Class-wise detection rate
        ax1 = axes[0, 0]
        class_stats = analysis_results.get('class_specific_stats', {})
        class_names = list(class_stats.keys())
        detection_rates = [stats.get('detection_rate', 0) for stats in class_stats.values()]
        
        bars = ax1.bar(class_names, detection_rates)
        # Color problematic classes differently
        for i, (name, bar) in enumerate(zip(class_names, bars)):
            if name in self.problematic_classes:
                bar.set_color('red')
            else:
                bar.set_color('green')
        
        ax1.set_title('Detection Rate by Class')
        ax1.set_ylabel('Detection Rate')
        ax1.set_ylim(0, 1.2)
        plt.setp(ax1.get_xticklabels(), rotation=45)
        
        # Add value labels on bars
        for bar, rate in zip(bars, detection_rates):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{rate:.3f}', ha='center', va='bottom')
        
        # Confidence distribution
        ax2 = axes[0, 1]
        quality_metrics = analysis_results.get('detection_quality_metrics', {})
        conf_calib = quality_metrics.get('confidence_calibration', {})
        
        conf_categories = ['High (>0.8)', 'Medium (0.5-0.8)', 'Low (<0.5)']
        conf_ratios = [
            conf_calib.get('high_confidence_ratio', 0),
            conf_calib.get('medium_confidence_ratio', 0),
            conf_calib.get('low_confidence_ratio', 0)
        ]
        
        ax2.pie(conf_ratios, labels=conf_categories, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Confidence Distribution')
        
        # Class-wise confidence
        ax3 = axes[0, 2]
        avg_confidences = [stats.get('avg_confidence', 0) for stats in class_stats.values()]
        bars3 = ax3.bar(class_names, avg_confidences)
        
        for i, (name, bar) in enumerate(zip(class_names, bars3)):
            if name in self.problematic_classes:
                bar.set_color('orange')
            else:
                bar.set_color('blue')
        
        ax3.set_title('Average Confidence by Class')
        ax3.set_ylabel('Average Confidence')
        ax3.set_ylim(0, 1)
        plt.setp(ax3.get_xticklabels(), rotation=45)
        
        # Object count comparison
        ax4 = axes[1, 0]
        pred_counts = [stats.get('prediction_count', 0) for stats in class_stats.values()]
        gt_counts = [stats.get('ground_truth_count', 0) for stats in class_stats.values()]
        
        x = np.arange(len(class_names))
        width = 0.35
        
        ax4.bar(x - width/2, pred_counts, width, label='Predictions', alpha=0.8)
        ax4.bar(x + width/2, gt_counts, width, label='Ground Truth', alpha=0.8)
        
        ax4.set_title('Predictions vs Ground Truth Count')
        ax4.set_ylabel('Count')
        ax4.set_xticks(x)
        ax4.set_xticklabels(class_names, rotation=45)
        ax4.legend()
        
        # Problem analysis for problematic classes
        ax5 = axes[1, 1]
        problematic_analysis = analysis_results.get('problematic_class_analysis', {})
        
        if problematic_analysis:
            problem_data = {}
            for class_name, analysis in problematic_analysis.items():
                problem_count = len(analysis.get('detection_challenges', [])) + \
                              len(analysis.get('failure_patterns', []))
                problem_data[class_name] = problem_count
            
            if problem_data:
                ax5.bar(problem_data.keys(), problem_data.values(), color='red', alpha=0.7)
                ax5.set_title('Issues Identified for Problematic Classes')
                ax5.set_ylabel('Number of Issues')
                plt.setp(ax5.get_xticklabels(), rotation=45)
        
        # Overall statistics
        ax6 = axes[1, 2]
        overall_stats = analysis_results.get('overall_stats', {})
        
        stats_text = f"Total Predictions: {overall_stats.get('total_predictions', 0)}\n"
        stats_text += f"Total Ground Truth: {overall_stats.get('total_ground_truth', 0)}\n"
        stats_text += f"Avg Confidence: {overall_stats.get('avg_confidence', 0):.3f}\n\n"
        
        # Add key issues
        stats_text += "Key Issues:\n"
        for class_name in self.problematic_classes:
            if class_name in class_stats:
                detection_rate = class_stats[class_name].get('detection_rate', 0)
                if detection_rate < 0.5:
                    stats_text += f"• Low {class_name} detection: {detection_rate:.3f}\n"
        
        ax6.text(0.1, 0.5, stats_text, transform=ax6.transAxes, fontsize=10,
                verticalalignment='center', bbox=dict(boxstyle="round", facecolor='wheat'))
        ax6.set_xlim(0, 1)
        ax6.set_ylim(0, 1)
        ax6.axis('off')
        ax6.set_title('Summary Statistics')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def update_detection_stats(self, predictions: Dict[str, Any], 
                             class_names: Optional[List[str]] = None):
        """Update internal detection statistics for monitoring"""
        if 'scores_3d' not in predictions or 'labels_3d' not in predictions:
            return
        
        scores = predictions['scores_3d']
        labels = predictions['labels_3d'] 
        bboxes = predictions.get('bboxes_3d', [])
        
        if hasattr(scores, 'numpy'):
            scores = scores.numpy()
        if hasattr(labels, 'numpy'):
            labels = labels.numpy()
        if hasattr(bboxes, 'numpy'):
            bboxes = bboxes.numpy()
        
        class_names = class_names or ['Pedestrian', 'Cyclist', 'Car']
        
        # Update overall stats
        self.detection_stats['total_detections'] += len(scores)
        
        # Update class-specific stats
        for score, label in zip(scores, labels):
            class_name = class_names[label] if label < len(class_names) else f'Class_{label}'
            self.detection_stats['class_detections'][class_name] += 1
            self.detection_stats['confidence_distributions'][class_name].append(score)
        
        # Update size and distance distributions if bboxes available
        if len(bboxes.shape) > 1 and bboxes.shape[1] >= 6:
            for bbox, label in zip(bboxes, labels):
                class_name = class_names[label] if label < len(class_names) else f'Class_{label}'
                
                # Size (volume)
                volume = bbox[3] * bbox[4] * bbox[5]  # w * h * l
                self.detection_stats['size_distributions'][class_name].append(volume)
                
                # Distance
                distance = np.sqrt(bbox[0]**2 + bbox[1]**2)
                self.detection_stats['distance_distributions'][class_name].append(distance)
    
    def get_detection_statistics_summary(self) -> Dict[str, Any]:
        """Get summary of accumulated detection statistics"""
        summary = {
            'total_detections': self.detection_stats['total_detections'],
            'class_distribution': dict(self.detection_stats['class_detections']),
            'confidence_stats': {},
            'size_stats': {},
            'distance_stats': {}
        }
        
        # Confidence statistics
        for class_name, confidences in self.detection_stats['confidence_distributions'].items():
            if confidences:
                summary['confidence_stats'][class_name] = {
                    'mean': np.mean(confidences),
                    'std': np.std(confidences),
                    'min': np.min(confidences),
                    'max': np.max(confidences),
                    'count': len(confidences)
                }
        
        # Size statistics
        for class_name, sizes in self.detection_stats['size_distributions'].items():
            if sizes:
                summary['size_stats'][class_name] = {
                    'mean_volume': np.mean(sizes),
                    'std_volume': np.std(sizes),
                    'small_object_ratio': np.sum(np.array(sizes) < np.percentile(sizes, 33)) / len(sizes)
                }
        
        # Distance statistics  
        for class_name, distances in self.detection_stats['distance_distributions'].items():
            if distances:
                summary['distance_stats'][class_name] = {
                    'mean_distance': np.mean(distances),
                    'near_object_ratio': np.sum(np.array(distances) < 20) / len(distances),
                    'far_object_ratio': np.sum(np.array(distances) > 50) / len(distances)
                }
        
        return summary