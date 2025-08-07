#pragma once

#include <vector>
#include <string>
#include <memory>
#include <Eigen/Dense>

namespace neurosim {

/**
 * @brief Vestibular system synthesis for balance and motion processing
 * 
 * This class simulates vestibular input processing, converting body motion,
 * balance, and spatial orientation into embeddings for neural simulation.
 * It includes:
 * - Linear acceleration detection (otoliths)
 * - Angular velocity detection (semicircular canals)
 * - Balance and postural control
 * - Spatial orientation and navigation
 * - Motion sickness and vestibular dysfunction
 * - Autism-specific vestibular processing differences
 * - PTSD-specific hypervigilance and startle responses
 */
class VestibularSynth {
public:
    /**
     * @brief Vestibular processing configuration
     */
    struct VestibularConfig {
        size_t embedding_dimension = 128;        ///< Output embedding size
        double sensitivity_threshold = 0.1;      ///< Motion detection threshold
        bool enable_linear_acceleration = true;  ///< Enable otolith simulation
        bool enable_angular_velocity = true;     ///< Enable semicircular canal simulation
        bool enable_balance_processing = true;   ///< Enable balance/postural processing
        double gravity_vector[3] = {0.0, 0.0, -9.81}; ///< Gravity reference vector
        
        // Autism-specific parameters
        bool autism_vestibular_differences = false; ///< Altered vestibular processing
        double autism_motion_sensitivity = 1.3;     ///< Enhanced motion sensitivity
        double autism_balance_difficulties = 1.2;   ///< Balance processing challenges
        bool autism_proprioceptive_differences = false; ///< Altered body awareness
        double autism_spatial_processing_variance = 1.1; ///< Spatial processing variability
        
        // PTSD-specific parameters
        bool ptsd_hypervigilance = false;        ///< Enhanced motion detection
        double ptsd_startle_sensitivity = 1.4;   ///< Enhanced startle to motion
        bool ptsd_balance_disruption = false;    ///< PTSD-related balance issues
        double ptsd_spatial_disorientation = 1.1; ///< Spatial disorientation tendency
        bool ptsd_freeze_response = false;       ///< Freeze response to motion
    };

    /**
     * @brief Vestibular input data structure
     */
    struct VestibularInput {
        // Linear motion (otoliths)
        double linear_acceleration[3] = {0.0, 0.0, 0.0}; ///< Linear acceleration (m/s²)
        double linear_velocity[3] = {0.0, 0.0, 0.0};     ///< Linear velocity (m/s)
        double position[3] = {0.0, 0.0, 0.0};            ///< Position (m)
        
        // Angular motion (semicircular canals)
        double angular_acceleration[3] = {0.0, 0.0, 0.0}; ///< Angular acceleration (rad/s²)
        double angular_velocity[3] = {0.0, 0.0, 0.0};     ///< Angular velocity (rad/s)
        double orientation[3] = {0.0, 0.0, 0.0};          ///< Orientation (roll, pitch, yaw)
        
        // Balance and postural information
        double center_of_pressure[2] = {0.0, 0.0};        ///< Center of pressure (x, y)
        double postural_sway = 0.0;                       ///< Postural sway magnitude
        bool feet_contact[2] = {true, true};              ///< Foot contact with ground
        
        double timestamp = 0.0;                           ///< Input timestamp
        std::string motion_context = "stationary";        ///< Motion context description
    };

    /**
     * @brief Vestibular processing result
     */
    struct VestibularEmbedding {
        Eigen::VectorXd feature_embedding;       ///< Main vestibular feature vector
        
        // Motion detection results
        double linear_motion_magnitude = 0.0;    ///< Overall linear motion intensity
        double angular_motion_magnitude = 0.0;   ///< Overall angular motion intensity
        std::string motion_type;                 ///< Classified motion type
        double motion_smoothness = 0.0;          ///< Motion smoothness/jerkiness
        
        // Balance and orientation
        double balance_stability = 1.0;          ///< Balance stability measure (0-1)
        double spatial_orientation_confidence = 1.0; ///< Orientation confidence
        std::string postural_state;              ///< Current postural state
        bool motion_sickness_risk = false;       ///< Motion sickness likelihood
        
        // Directional information
        std::string primary_motion_direction;    ///< Primary direction of motion
        double motion_predictability = 0.0;      ///< How predictable the motion is
        bool sudden_motion_detected = false;     ///< Sudden motion change detected
        
        // Autism-specific metrics
        struct {
            double motion_hypersensitivity = 0.0; ///< Motion sensitivity level
            double balance_difficulty = 0.0;      ///< Balance processing difficulty
            double proprioceptive_uncertainty = 0.0; ///< Body position uncertainty
            bool sensory_seeking_motion = false;  ///< Motion-seeking behavior
            double spatial_processing_load = 0.0; ///< Spatial processing demand
        } autism_metrics;
        
        // PTSD-specific metrics
        struct {
            double hypervigilance_activation = 0.0; ///< Motion hypervigilance level
            bool startle_trigger = false;           ///< Motion-triggered startle
            double spatial_disorientation = 0.0;    ///< Disorientation level
            bool freeze_response_triggered = false; ///< Freeze response activation
            double threat_motion_detected = 0.0;    ///< Threatening motion level
        } ptsd_metrics;
        
        double processing_confidence = 1.0;       ///< Confidence in processing result
        double processing_time_ms = 0.0;          ///< Simulated processing time
    };

public:
    /**
     * @brief Constructor
     * @param config Vestibular processing configuration
     */
    explicit VestibularSynth(const VestibularConfig& config = VestibularConfig{});

    /**
     * @brief Process vestibular input and generate embedding
     * @param input Vestibular input data
     * @return Vestibular embedding result
     */
    VestibularEmbedding processVestibularInput(const VestibularInput& input);

    /**
     * @brief Process simulated motion scenario
     * @param motion_description Text description of motion/balance scenario
     * @return Simulated vestibular embedding
     */
    VestibularEmbedding processSimulatedMotion(const std::string& motion_description);

    /**
     * @brief Simulate specific motion patterns
     * @param motion_type Type of motion ("walking", "running", "turning", etc.)
     * @param intensity Motion intensity (0-1)
     * @param duration Duration in seconds
     * @return Vestibular embedding for the motion
     */
    VestibularEmbedding simulateMotionPattern(const std::string& motion_type, 
                                            double intensity, double duration);

    /**
     * @brief Update processing configuration
     * @param config New configuration
     */
    void updateConfig(const VestibularConfig& config);

    /**
     * @brief Get current configuration
     * @return Current vestibular config
     */
    const VestibularConfig& getConfig() const { return config_; }

    /**
     * @brief Calibrate vestibular system (set baseline)
     * @param baseline_input Baseline vestibular state
     */
    void calibrateBaseline(const VestibularInput& baseline_input);

    /**
     * @brief Get processing history for analysis
     * @return Vector of recent vestibular processing results
     */
    std::vector<VestibularEmbedding> getProcessingHistory() const;

    /**
     * @brief Clear processing history
     */
    void clearHistory();

private:
    VestibularConfig config_;
    VestibularInput baseline_state_;
    std::vector<VestibularEmbedding> processing_history_;
    
    // Core vestibular processing methods
    Eigen::VectorXd extractVestibularFeatures(const VestibularInput& input);
    double calculateLinearMotionMagnitude(const VestibularInput& input);
    double calculateAngularMotionMagnitude(const VestibularInput& input);
    std::string classifyMotionType(const VestibularInput& input);
    double calculateMotionSmoothness(const VestibularInput& input);
    
    // Balance and orientation processing
    double calculateBalanceStability(const VestibularInput& input);
    double calculateSpatialOrientationConfidence(const VestibularInput& input);
    std::string assessPosturalState(const VestibularInput& input);
    bool assessMotionSicknessRisk(const VestibularInput& input);
    
    // Motion analysis
    std::string determinePrimaryMotionDirection(const VestibularInput& input);
    double calculateMotionPredictability(const VestibularInput& input);
    bool detectSuddenMotion(const VestibularInput& input);
    
    // Autism-specific processing
    void applyAutismVestibularProcessing(VestibularEmbedding& result, const VestibularInput& input);
    double calculateMotionHypersensitivity(const VestibularInput& input);
    double calculateBalanceDifficulty(const VestibularInput& input);
    double calculateProprioceptiveUncertainty(const VestibularInput& input);
    bool checkSensorySeekingMotion(const VestibularInput& input);
    double calculateSpatialProcessingLoad(const VestibularInput& input);
    
    // PTSD-specific processing
    void applyPTSDVestibularProcessing(VestibularEmbedding& result, const VestibularInput& input);
    double calculateHypervigilanceActivation(const VestibularInput& input);
    bool checkStartleTrigger(const VestibularInput& input);
    double calculateSpatialDisorientation(const VestibularInput& input);
    bool checkFreezeResponse(const VestibularInput& input);
    double calculateThreatMotion(const VestibularInput& input);
    
    // Utility methods
    double calculateProcessingTime(const VestibularInput& input);
    double calculateProcessingConfidence(const VestibularEmbedding& result);
    Eigen::VectorXd normalizeFeatures(const Eigen::VectorXd& features);
    double calculateMagnitude(const double vector[3]);
    double calculateAngleBetweenVectors(const double v1[3], const double v2[3]);
    
    // Simulated vestibular processing
    VestibularEmbedding simulateVestibularProcessing(const std::string& description);
    VestibularInput generateMotionPattern(const std::string& motion_type, double intensity);
    std::vector<std::string> parseMotionDescription(const std::string& description);
    
    // Static data for motion classification
    static const std::vector<std::string> motion_types_;
    static const std::vector<std::string> postural_states_;
    static const std::vector<std::string> motion_directions_;
};

} // namespace neurosim
