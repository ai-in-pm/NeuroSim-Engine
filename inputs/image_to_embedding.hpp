#pragma once

#include <vector>
#include <string>
#include <memory>
#include <Eigen/Dense>

namespace neurosim {

/**
 * @brief Visual embedding generator for image processing
 * 
 * This class simulates visual feature extraction similar to CLIP or other
 * vision transformers, converting visual input into embeddings that can
 * be processed by the neural simulation. It includes:
 * - Basic visual feature extraction
 * - Object detection and classification
 * - Facial expression recognition
 * - Scene understanding
 * - Autism-specific visual processing differences
 * - PTSD-specific threat detection in visual stimuli
 */
class ImageToEmbedding {
public:
    /**
     * @brief Visual processing configuration
     */
    struct VisualConfig {
        size_t embedding_dimension = 512;        ///< Output embedding size
        double feature_extraction_strength = 1.0; ///< Feature extraction intensity
        bool enable_face_detection = true;       ///< Enable facial processing
        bool enable_object_detection = true;     ///< Enable object recognition
        bool enable_scene_analysis = true;       ///< Enable scene understanding
        
        // Autism-specific parameters
        bool autism_detail_focus = false;        ///< Enhanced detail processing
        double autism_face_processing_deficit = 0.6; ///< Reduced face processing
        double autism_pattern_enhancement = 1.4; ///< Enhanced pattern detection
        bool autism_peripheral_hypersensitivity = false; ///< Enhanced peripheral vision
        
        // PTSD-specific parameters
        bool ptsd_threat_hypervigilance = false; ///< Enhanced threat detection
        double ptsd_startle_sensitivity = 1.5;   ///< Enhanced startle to visual stimuli
        std::vector<std::string> ptsd_trigger_objects; ///< Objects that trigger PTSD
        bool ptsd_hyperscanning = false;         ///< Continuous threat scanning
    };

    /**
     * @brief Visual input data structure
     */
    struct VisualInput {
        std::vector<uint8_t> image_data;         ///< Raw image data (RGB)
        size_t width = 0;                        ///< Image width
        size_t height = 0;                       ///< Image height
        size_t channels = 3;                     ///< Number of channels (RGB=3)
        double timestamp = 0.0;                  ///< Input timestamp
        std::string image_format = "RGB";        ///< Image format
        double brightness = 0.5;                 ///< Normalized brightness (0-1)
        double contrast = 0.5;                   ///< Normalized contrast (0-1)
    };

    /**
     * @brief Visual processing result
     */
    struct VisualEmbedding {
        Eigen::VectorXd feature_embedding;       ///< Main visual feature vector
        std::vector<std::string> detected_objects; ///< Detected objects/entities
        std::vector<std::string> detected_faces;  ///< Detected facial expressions
        std::string scene_category;              ///< Scene classification
        double visual_complexity = 0.0;          ///< Scene complexity measure
        
        // Spatial attention map (simplified)
        std::vector<double> attention_weights;    ///< Spatial attention distribution
        
        // Autism-specific metrics
        struct {
            double detail_saliency = 0.0;        ///< Detail-focused attention
            std::vector<std::string> pattern_features; ///< Detected patterns
            double face_processing_confidence = 0.0; ///< Face processing quality
            bool peripheral_activation = false;   ///< Peripheral vision activation
        } autism_metrics;
        
        // PTSD-specific metrics
        struct {
            double threat_level = 0.0;            ///< Detected threat level
            std::vector<std::string> threat_objects; ///< Threatening objects detected
            bool startle_trigger = false;         ///< Whether visual startle occurred
            double hypervigilance_activation = 0.0; ///< Hypervigilance level
        } ptsd_metrics;
        
        double processing_confidence = 1.0;       ///< Confidence in processing result
        double processing_time_ms = 0.0;          ///< Simulated processing time
    };

public:
    /**
     * @brief Constructor
     * @param config Visual processing configuration
     */
    explicit ImageToEmbedding(const VisualConfig& config = VisualConfig{});

    /**
     * @brief Process visual input and generate embedding
     * @param input Visual input data
     * @return Visual embedding result
     */
    VisualEmbedding processImage(const VisualInput& input);

    /**
     * @brief Process image from file path (convenience method)
     * @param image_path Path to image file
     * @return Visual embedding result
     */
    VisualEmbedding processImageFile(const std::string& image_path);

    /**
     * @brief Process simulated visual scene
     * @param scene_description Text description of visual scene
     * @return Simulated visual embedding
     */
    VisualEmbedding processSimulatedScene(const std::string& scene_description);

    /**
     * @brief Update processing configuration
     * @param config New configuration
     */
    void updateConfig(const VisualConfig& config);

    /**
     * @brief Get current configuration
     * @return Current visual config
     */
    const VisualConfig& getConfig() const { return config_; }

    /**
     * @brief Add PTSD trigger object for threat detection
     * @param object_name Object that triggers PTSD response
     * @param threat_level Threat level (0-1)
     */
    void addPTSDTriggerObject(const std::string& object_name, double threat_level = 0.8);

    /**
     * @brief Get processing history for analysis
     * @return Vector of recent visual processing results
     */
    std::vector<VisualEmbedding> getProcessingHistory() const;

    /**
     * @brief Clear processing history
     */
    void clearHistory();

private:
    VisualConfig config_;
    std::vector<VisualEmbedding> processing_history_;
    
    // Core visual processing methods
    Eigen::VectorXd extractBasicFeatures(const VisualInput& input);
    std::vector<std::string> detectObjects(const VisualInput& input);
    std::vector<std::string> detectFaces(const VisualInput& input);
    std::string classifyScene(const VisualInput& input);
    double calculateVisualComplexity(const VisualInput& input);
    std::vector<double> generateAttentionMap(const VisualInput& input);
    
    // Autism-specific processing
    void applyAutismVisualProcessing(VisualEmbedding& result, const VisualInput& input);
    double calculateDetailSaliency(const VisualInput& input);
    std::vector<std::string> detectPatterns(const VisualInput& input);
    double processFacesWithAutism(const VisualInput& input);
    bool checkPeripheralActivation(const VisualInput& input);
    
    // PTSD-specific processing
    void applyPTSDVisualProcessing(VisualEmbedding& result, const VisualInput& input);
    double calculateThreatLevel(const VisualInput& input, const std::vector<std::string>& objects);
    std::vector<std::string> identifyThreatObjects(const std::vector<std::string>& objects);
    bool checkStartleTrigger(const VisualInput& input);
    double calculateHypervigilanceActivation(const VisualInput& input);
    
    // Utility methods
    double calculateProcessingTime(const VisualInput& input);
    double calculateProcessingConfidence(const VisualEmbedding& result);
    Eigen::VectorXd normalizeFeatures(const Eigen::VectorXd& features);
    
    // Simulated visual processing (for when no actual image processing is available)
    VisualEmbedding simulateVisualProcessing(const std::string& description);
    std::vector<std::string> parseSceneDescription(const std::string& description);
    double estimateComplexityFromDescription(const std::string& description);
    
    // Static data for object/scene recognition
    static const std::vector<std::string> common_objects_;
    static const std::vector<std::string> facial_expressions_;
    static const std::vector<std::string> scene_categories_;
    static const std::vector<std::string> threat_objects_;
};

} // namespace neurosim
