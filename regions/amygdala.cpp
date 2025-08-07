#include "amygdala.hpp"
#include <algorithm>
#include <random>

namespace neurosim {

Amygdala::Amygdala(const RegionConfig& region_config, const AmygdalaConfig& amygdala_config)
    : BrainRegion(region_config), amygdala_config_(amygdala_config) {
    
    // Initialize amygdala-specific state
    amygdala_state_.threat_level = 0.0;
    amygdala_state_.emotional_arousal = 0.0;
    amygdala_state_.fear_response = 0.0;
    amygdala_state_.social_anxiety = 0.0;
    amygdala_state_.habituation_level = 0.0;
    amygdala_state_.sensitization_level = 0.0;
}

double Amygdala::processInput(double input, double dt) {
    current_time_ += dt;
    
    // Process input through microcircuit
    auto microcircuit_state = microcircuit_->process(input, dt);
    
    // Calculate threat level from input
    Eigen::VectorXd input_vector = Eigen::VectorXd::Constant(1, input);
    amygdala_state_.threat_level = calculateThreatLevel(input_vector);
    
    // Calculate emotional arousal
    amygdala_state_.emotional_arousal = calculateEmotionalArousal(
        amygdala_state_.threat_level, input);
    
    // Calculate fear response
    amygdala_state_.fear_response = amygdala_state_.threat_level * 
                                   amygdala_state_.emotional_arousal;
    
    // Apply habituation effect
    double habituated_activation = applyHabituationEffect(amygdala_state_.fear_response);
    
    // Apply sensitization effect
    double sensitized_activation = applySensitizationEffect(habituated_activation);
    
    // Update habituation and sensitization
    updateHabituation(input, dt);
    updateSensitization(amygdala_state_.threat_level, dt);
    
    // Apply autism modifications if enabled
    if (amygdala_config_.autism_social_hypersensitivity) {
        applyAutismModifications(sensitized_activation, input_vector);
    }
    
    // Apply PTSD modifications if enabled
    if (amygdala_config_.ptsd_hypervigilance) {
        applyPTSDModifications(sensitized_activation, input_vector);
    }
    
    // Check for fight-or-flight activation
    amygdala_state_.fight_flight_active = sensitized_activation > 0.7;
    
    // Check for memory consolidation
    amygdala_state_.memory_consolidation_active = 
        amygdala_state_.emotional_arousal > 0.5;
    
    // Update current activation
    current_activation_ = std::max(0.0, std::min(1.0, sensitized_activation));
    
    return current_activation_;
}

double Amygdala::processThreatAssessment(const Eigen::VectorXd& visual_input,
                                        const Eigen::VectorXd& auditory_input,
                                        const Eigen::VectorXd& social_context,
                                        double dt) {
    
    // Combine multi-modal threat cues
    double visual_threat = calculateThreatLevel(visual_input);
    double auditory_threat = calculateThreatLevel(auditory_input) * 1.2; // Auditory bias
    double social_threat = calculateSocialThreat(social_context);
    
    // Weighted combination
    double combined_threat = visual_threat * 0.4 + auditory_threat * 0.4 + social_threat * 0.2;
    
    // Apply autism social hypersensitivity
    if (amygdala_config_.autism_social_hypersensitivity) {
        combined_threat += social_threat * 0.5; // Enhanced social threat sensitivity
    }
    
    // Apply PTSD hypervigilance
    if (amygdala_config_.ptsd_hypervigilance) {
        combined_threat *= amygdala_config_.ptsd_trauma_sensitivity;
    }
    
    amygdala_state_.threat_level = std::min(1.0, combined_threat);
    
    return amygdala_state_.threat_level;
}

void Amygdala::processMemoryConsolidation(double emotional_valence, 
                                         const Eigen::VectorXd& memory_content,
                                         double dt) {
    
    // Only consolidate if emotional arousal is sufficient
    if (amygdala_state_.emotional_arousal > 0.3) {
        updateEmotionalMemories(emotional_valence, memory_content);
        amygdala_state_.memory_consolidation_active = true;
    } else {
        amygdala_state_.memory_consolidation_active = false;
    }
}

double Amygdala::checkTraumaActivation(const Eigen::VectorXd& input_pattern) {
    double max_match = 0.0;
    
    for (const auto& trauma_template : amygdala_config_.trauma_templates) {
        double match_strength = calculateMemoryMatch(input_pattern, trauma_template);
        max_match = std::max(max_match, match_strength);
        
        if (match_strength > 0.7) { // Threshold for trauma activation
            amygdala_state_.trauma_flashback_triggered = true;
            amygdala_state_.emotional_arousal = std::min(1.0, 
                amygdala_state_.emotional_arousal + match_strength * 0.5);
        }
    }
    
    return max_match;
}

void Amygdala::addTraumaTemplate(const Eigen::VectorXd& trauma_pattern, double sensitivity) {
    amygdala_config_.trauma_templates.push_back(trauma_pattern);
    // Note: In a full implementation, we'd store sensitivity with each template
}

double Amygdala::calculateThreatLevel(const Eigen::VectorXd& input) const {
    if (input.size() == 0) return 0.0;
    
    // Simple threat calculation based on input magnitude and characteristics
    double magnitude = input.norm();
    
    // Apply threat sensitivity
    double threat = magnitude * amygdala_config_.threat_sensitivity;
    
    // Add some randomness for variability
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<> noise_dist(-0.1, 0.1);
    
    threat += noise_dist(gen);
    
    return std::max(0.0, std::min(1.0, threat));
}

double Amygdala::calculateSocialThreat(const Eigen::VectorXd& social_context) const {
    if (social_context.size() == 0) return 0.0;
    
    double social_magnitude = social_context.norm();
    double social_threat = social_magnitude * amygdala_config_.social_threat_bias;
    
    // Apply autism social hypersensitivity
    if (amygdala_config_.autism_social_hypersensitivity) {
        social_threat *= amygdala_config_.autism_threat_generalization;
    }
    
    return std::max(0.0, std::min(1.0, social_threat));
}

double Amygdala::calculateEmotionalArousal(double threat_level, double input_strength) const {
    // Emotional arousal increases with threat and input intensity
    double arousal = threat_level * 0.7 + input_strength * 0.3;
    
    // Apply autism emotional dysregulation
    if (amygdala_config_.autism_social_hypersensitivity) {
        arousal *= amygdala_config_.autism_emotional_dysregulation;
    }
    
    // Apply PTSD emotional dysregulation
    if (amygdala_config_.ptsd_hypervigilance) {
        arousal *= amygdala_config_.ptsd_emotional_dysregulation;
    }
    
    return std::max(0.0, std::min(1.0, arousal));
}

void Amygdala::applyAutismModifications(double& activation, const Eigen::VectorXd& input) {
    // Enhanced threat generalization
    activation *= amygdala_config_.autism_threat_generalization;
    
    // Calculate social anxiety for autism
    amygdala_state_.social_anxiety = calculateAutismSocialAnxiety(input);
    
    // Reduced habituation in autism
    amygdala_state_.habituation_level *= 0.7;
}

void Amygdala::applyPTSDModifications(double& activation, const Eigen::VectorXd& input) {
    // Enhanced trauma sensitivity
    activation *= amygdala_config_.ptsd_trauma_sensitivity;
    
    // Check for memory intrusion
    if (checkMemoryIntrusion(input)) {
        amygdala_state_.trauma_flashback_triggered = true;
        activation = std::min(1.0, activation + 0.5); // Boost activation during flashback
    }
    
    // Reduced habituation in PTSD
    amygdala_state_.habituation_level *= 0.5;
}

double Amygdala::calculateAutismSocialAnxiety(const Eigen::VectorXd& social_context) const {
    if (social_context.size() == 0) return 0.0;
    
    // Social anxiety increases with social complexity
    double social_complexity = social_context.norm();
    return std::min(1.0, social_complexity * 1.5); // Enhanced in autism
}

bool Amygdala::checkMemoryIntrusion(const Eigen::VectorXd& input) const {
    // Check if current input matches stored trauma patterns
    for (const auto& trauma_template : amygdala_config_.trauma_templates) {
        double match = calculateMemoryMatch(input, trauma_template);
        if (match > 0.6) { // Lower threshold for PTSD intrusion
            return true;
        }
    }
    return false;
}

void Amygdala::updateEmotionalMemories(double emotional_valence, 
                                      const Eigen::VectorXd& memory_content) {
    // Store emotional memory with valence
    emotional_memories_.emplace_back(memory_content, emotional_valence);
    
    // Limit memory storage
    if (emotional_memories_.size() > 1000) {
        emotional_memories_.erase(emotional_memories_.begin());
    }
}

double Amygdala::calculateMemoryMatch(const Eigen::VectorXd& input, 
                                     const Eigen::VectorXd& stored_pattern) const {
    if (input.size() == 0 || stored_pattern.size() == 0) return 0.0;
    
    // Calculate cosine similarity
    double dot_product = input.dot(stored_pattern);
    double input_norm = input.norm();
    double pattern_norm = stored_pattern.norm();
    
    if (input_norm == 0.0 || pattern_norm == 0.0) return 0.0;
    
    return std::max(0.0, dot_product / (input_norm * pattern_norm));
}

void Amygdala::updateHabituation(double input_strength, double dt) {
    // Habituation increases with repeated exposure
    double habituation_increment = input_strength * amygdala_config_.habituation_rate * dt / 1000.0;
    amygdala_state_.habituation_level = std::min(1.0, 
        amygdala_state_.habituation_level + habituation_increment);
    
    // Habituation decays over time without stimulation
    if (input_strength < 0.1) {
        amygdala_state_.habituation_level *= 0.999; // Slow decay
    }
}

void Amygdala::updateSensitization(double threat_level, double dt) {
    // Sensitization increases with high threat levels
    if (threat_level > 0.7) {
        double sensitization_increment = threat_level * 0.01 * dt / 1000.0;
        amygdala_state_.sensitization_level = std::min(1.0, 
            amygdala_state_.sensitization_level + sensitization_increment);
    } else {
        // Sensitization decays slowly
        amygdala_state_.sensitization_level *= 0.9995;
    }
}

double Amygdala::applyHabituationEffect(double base_activation) const {
    // Habituation reduces response to repeated stimuli
    return base_activation * (1.0 - amygdala_state_.habituation_level * 0.5);
}

double Amygdala::applySensitizationEffect(double base_activation) const {
    // Sensitization enhances response to stimuli
    return base_activation * (1.0 + amygdala_state_.sensitization_level * 0.3);
}

std::vector<std::string> Amygdala::identifyThreats(const Eigen::VectorXd& input) const {
    std::vector<std::string> threats;
    
    // Simple threat identification based on input characteristics
    if (input.norm() > 0.7) {
        threats.push_back("high_intensity_stimulus");
    }
    
    if (amygdala_state_.trauma_flashback_triggered) {
        threats.push_back("trauma_trigger");
    }
    
    if (amygdala_state_.social_anxiety > 0.6) {
        threats.push_back("social_threat");
    }
    
    return threats;
}

void Amygdala::updateActiveMemories(const Eigen::VectorXd& input) {
    amygdala_state_.active_memories.clear();
    
    // Check which stored memories are activated by current input
    for (size_t i = 0; i < emotional_memories_.size(); ++i) {
        double match = calculateMemoryMatch(input, emotional_memories_[i].first);
        if (match > 0.5) {
            amygdala_state_.active_memories.push_back("memory_" + std::to_string(i));
        }
    }
    
    // Update detected threats
    amygdala_state_.detected_threats = identifyThreats(input);
}

void Amygdala::updateConfig(const AmygdalaConfig& config) {
    amygdala_config_ = config;
}

std::vector<std::pair<Eigen::VectorXd, double>> Amygdala::getEmotionalMemories() const {
    return emotional_memories_;
}

} // namespace neurosim
