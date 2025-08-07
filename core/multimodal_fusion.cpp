#include "multimodal_fusion.hpp"
#include <algorithm>
#include <numeric>
#include <random>

namespace neurosim {

MultiModalFusion::MultiModalFusion(const FusionConfig& config) : config_(config) {
}

MultiModalFusion::FusedRepresentation MultiModalFusion::fuse(const SensoryInput& input) {
    FusedRepresentation result;
    
    // Calculate modality contributions
    result.modality_contributions = calculateModalityContributions(input);
    
    // Perform weighted fusion
    result.unified_embedding = performWeightedFusion(input);
    
    // Calculate fusion confidence
    result.fusion_confidence = calculateFusionConfidence(input, result.unified_embedding);
    
    // Calculate sensory overload
    result.sensory_overload = calculateSensoryOverload(input);
    
    // Set fusion metadata
    result.fusion_metadata.dominant_modality = identifyDominantModality(result.modality_contributions);
    result.fusion_metadata.cross_modal_conflict = calculateCrossModalConflict(input);
    result.fusion_metadata.sensory_gating_active = applySensoryGating(input);
    
    // Apply autism-specific processing
    if (config_.autism_sensory_hypersensitivity) {
        applyAutismProcessing(result, input);
    }
    
    // Apply PTSD-specific processing
    if (config_.ptsd_hypervigilance) {
        applyPTSDProcessing(result, input);
    }
    
    // Store in history
    fusion_history_.push_back(result);
    if (fusion_history_.size() > 1000) {
        fusion_history_.erase(fusion_history_.begin());
    }
    
    return result;
}

MultiModalFusion::FusedRepresentation MultiModalFusion::fuseTemporalSequence(
    const std::vector<SensoryInput>& inputs) {
    
    if (inputs.empty()) {
        return FusedRepresentation{};
    }
    
    // Update temporal buffer
    for (const auto& input : inputs) {
        updateTemporalBuffer(input);
    }
    
    // Perform temporal integration
    Eigen::VectorXd temporal_embedding = performTemporalIntegration(inputs);
    
    // Use the latest input for other processing
    const auto& latest_input = inputs.back();
    auto result = fuse(latest_input);
    
    // Replace unified embedding with temporally integrated version
    result.unified_embedding = temporal_embedding;
    
    return result;
}

Eigen::VectorXd MultiModalFusion::performWeightedFusion(const SensoryInput& input) const {
    // Determine the size of the unified embedding (use largest modality)
    size_t max_size = 0;
    if (input.visual.size() > 0) max_size = std::max(max_size, static_cast<size_t>(input.visual.size()));
    if (input.auditory.size() > 0) max_size = std::max(max_size, static_cast<size_t>(input.auditory.size()));
    if (input.vestibular.size() > 0) max_size = std::max(max_size, static_cast<size_t>(input.vestibular.size()));
    if (input.interoceptive.size() > 0) max_size = std::max(max_size, static_cast<size_t>(input.interoceptive.size()));
    
    if (max_size == 0) {
        return Eigen::VectorXd::Zero(512); // Default size
    }
    
    Eigen::VectorXd fused_embedding = Eigen::VectorXd::Zero(max_size);
    
    // Resize and weight each modality
    if (input.visual.size() > 0) {
        Eigen::VectorXd visual_resized = Eigen::VectorXd::Zero(max_size);
        visual_resized.head(std::min(max_size, static_cast<size_t>(input.visual.size()))) = 
            input.visual.head(std::min(static_cast<int>(max_size), input.visual.size()));
        fused_embedding += config_.visual_weight * visual_resized;
    }
    
    if (input.auditory.size() > 0) {
        Eigen::VectorXd auditory_resized = Eigen::VectorXd::Zero(max_size);
        auditory_resized.head(std::min(max_size, static_cast<size_t>(input.auditory.size()))) = 
            input.auditory.head(std::min(static_cast<int>(max_size), input.auditory.size()));
        fused_embedding += config_.auditory_weight * auditory_resized;
    }
    
    if (input.vestibular.size() > 0) {
        Eigen::VectorXd vestibular_resized = Eigen::VectorXd::Zero(max_size);
        vestibular_resized.head(std::min(max_size, static_cast<size_t>(input.vestibular.size()))) = 
            input.vestibular.head(std::min(static_cast<int>(max_size), input.vestibular.size()));
        fused_embedding += config_.vestibular_weight * vestibular_resized;
    }
    
    if (input.interoceptive.size() > 0) {
        Eigen::VectorXd interoceptive_resized = Eigen::VectorXd::Zero(max_size);
        interoceptive_resized.head(std::min(max_size, static_cast<size_t>(input.interoceptive.size()))) = 
            input.interoceptive.head(std::min(static_cast<int>(max_size), input.interoceptive.size()));
        fused_embedding += config_.interoceptive_weight * interoceptive_resized;
    }
    
    return normalizeEmbedding(fused_embedding);
}

std::vector<double> MultiModalFusion::calculateModalityContributions(const SensoryInput& input) const {
    std::vector<double> contributions(4, 0.0); // visual, auditory, vestibular, interoceptive
    
    double total_magnitude = 0.0;
    
    if (input.visual.size() > 0) {
        contributions[0] = calculateEmbeddingMagnitude(input.visual) * config_.visual_weight;
        total_magnitude += contributions[0];
    }
    
    if (input.auditory.size() > 0) {
        contributions[1] = calculateEmbeddingMagnitude(input.auditory) * config_.auditory_weight;
        total_magnitude += contributions[1];
    }
    
    if (input.vestibular.size() > 0) {
        contributions[2] = calculateEmbeddingMagnitude(input.vestibular) * config_.vestibular_weight;
        total_magnitude += contributions[2];
    }
    
    if (input.interoceptive.size() > 0) {
        contributions[3] = calculateEmbeddingMagnitude(input.interoceptive) * config_.interoceptive_weight;
        total_magnitude += contributions[3];
    }
    
    // Normalize contributions
    if (total_magnitude > 0.0) {
        for (auto& contrib : contributions) {
            contrib /= total_magnitude;
        }
    }
    
    return contributions;
}

double MultiModalFusion::calculateFusionConfidence(const SensoryInput& input, 
                                                  const Eigen::VectorXd& fused_embedding) const {
    // Simple confidence based on input quality and consistency
    double confidence = input.confidence;
    
    // Reduce confidence if there's high cross-modal conflict
    double conflict = calculateCrossModalConflict(input);
    confidence *= (1.0 - conflict * 0.5);
    
    // Reduce confidence if sensory overload is high
    double overload = calculateSensoryOverload(input);
    confidence *= (1.0 - overload * 0.3);
    
    return std::max(0.0, std::min(1.0, confidence));
}

double MultiModalFusion::calculateSensoryOverload(const SensoryInput& input) const {
    double total_intensity = 0.0;
    int modality_count = 0;
    
    if (input.visual.size() > 0) {
        total_intensity += calculateEmbeddingMagnitude(input.visual);
        modality_count++;
    }
    
    if (input.auditory.size() > 0) {
        total_intensity += calculateEmbeddingMagnitude(input.auditory);
        modality_count++;
    }
    
    if (input.vestibular.size() > 0) {
        total_intensity += calculateEmbeddingMagnitude(input.vestibular);
        modality_count++;
    }
    
    if (input.interoceptive.size() > 0) {
        total_intensity += calculateEmbeddingMagnitude(input.interoceptive);
        modality_count++;
    }
    
    if (modality_count == 0) return 0.0;
    
    double average_intensity = total_intensity / modality_count;
    
    // Apply autism sensory hypersensitivity
    if (config_.autism_sensory_hypersensitivity) {
        average_intensity *= 1.5;
    }
    
    return std::min(1.0, average_intensity);
}

std::string MultiModalFusion::identifyDominantModality(const std::vector<double>& contributions) const {
    if (contributions.size() < 4) return "unknown";
    
    auto max_it = std::max_element(contributions.begin(), contributions.end());
    size_t max_index = std::distance(contributions.begin(), max_it);
    
    switch (max_index) {
        case 0: return "visual";
        case 1: return "auditory";
        case 2: return "vestibular";
        case 3: return "interoceptive";
        default: return "unknown";
    }
}

double MultiModalFusion::calculateCrossModalConflict(const SensoryInput& input) const {
    // Simplified conflict calculation based on magnitude differences
    std::vector<double> magnitudes;
    
    if (input.visual.size() > 0) magnitudes.push_back(calculateEmbeddingMagnitude(input.visual));
    if (input.auditory.size() > 0) magnitudes.push_back(calculateEmbeddingMagnitude(input.auditory));
    if (input.vestibular.size() > 0) magnitudes.push_back(calculateEmbeddingMagnitude(input.vestibular));
    if (input.interoceptive.size() > 0) magnitudes.push_back(calculateEmbeddingMagnitude(input.interoceptive));
    
    if (magnitudes.size() < 2) return 0.0;
    
    double mean = std::accumulate(magnitudes.begin(), magnitudes.end(), 0.0) / magnitudes.size();
    double variance = 0.0;
    
    for (double mag : magnitudes) {
        variance += (mag - mean) * (mag - mean);
    }
    variance /= magnitudes.size();
    
    return std::min(1.0, variance);
}

bool MultiModalFusion::applySensoryGating(const SensoryInput& input) const {
    double overload = calculateSensoryOverload(input);
    return overload > config_.sensory_gating_threshold;
}

void MultiModalFusion::applyAutismProcessing(FusedRepresentation& result, const SensoryInput& input) const {
    result.autism_metrics.hypersensitivity_activation = calculateHypersensitivityActivation(input);
    result.autism_metrics.overwhelming_modalities = identifyOverwhelmingModalities(input);
    
    // Enhance sensory overload in autism
    result.sensory_overload *= 1.3;
}

void MultiModalFusion::applyPTSDProcessing(FusedRepresentation& result, const SensoryInput& input) const {
    result.ptsd_metrics.threat_salience = calculateThreatSalience(input);
    result.ptsd_metrics.trigger_modalities = identifyTriggerModalities(input);
}

double MultiModalFusion::calculateHypersensitivityActivation(const SensoryInput& input) const {
    double max_intensity = 0.0;
    
    if (input.visual.size() > 0) {
        max_intensity = std::max(max_intensity, calculateEmbeddingMagnitude(input.visual));
    }
    if (input.auditory.size() > 0) {
        max_intensity = std::max(max_intensity, calculateEmbeddingMagnitude(input.auditory));
    }
    if (input.vestibular.size() > 0) {
        max_intensity = std::max(max_intensity, calculateEmbeddingMagnitude(input.vestibular));
    }
    if (input.interoceptive.size() > 0) {
        max_intensity = std::max(max_intensity, calculateEmbeddingMagnitude(input.interoceptive));
    }
    
    return std::min(1.0, max_intensity * 1.5); // Enhanced in autism
}

std::vector<std::string> MultiModalFusion::identifyOverwhelmingModalities(const SensoryInput& input) const {
    std::vector<std::string> overwhelming;
    double threshold = 0.7;
    
    if (input.visual.size() > 0 && calculateEmbeddingMagnitude(input.visual) > threshold) {
        overwhelming.push_back("visual");
    }
    if (input.auditory.size() > 0 && calculateEmbeddingMagnitude(input.auditory) > threshold) {
        overwhelming.push_back("auditory");
    }
    if (input.vestibular.size() > 0 && calculateEmbeddingMagnitude(input.vestibular) > threshold) {
        overwhelming.push_back("vestibular");
    }
    if (input.interoceptive.size() > 0 && calculateEmbeddingMagnitude(input.interoceptive) > threshold) {
        overwhelming.push_back("interoceptive");
    }
    
    return overwhelming;
}

double MultiModalFusion::calculateThreatSalience(const SensoryInput& input) const {
    // Simple threat detection based on high-intensity, sudden changes
    double threat_score = 0.0;
    
    // High auditory intensity might indicate threat
    if (input.auditory.size() > 0) {
        threat_score += calculateEmbeddingMagnitude(input.auditory) * 0.4;
    }
    
    // High vestibular activity might indicate threat
    if (input.vestibular.size() > 0) {
        threat_score += calculateEmbeddingMagnitude(input.vestibular) * 0.3;
    }
    
    // High interoceptive arousal might indicate threat
    if (input.interoceptive.size() > 0) {
        threat_score += calculateEmbeddingMagnitude(input.interoceptive) * 0.3;
    }
    
    return std::min(1.0, threat_score);
}

std::vector<std::string> MultiModalFusion::identifyTriggerModalities(const SensoryInput& input) const {
    std::vector<std::string> triggers;
    double threat_threshold = 0.6;
    
    if (input.auditory.size() > 0 && calculateEmbeddingMagnitude(input.auditory) > threat_threshold) {
        triggers.push_back("auditory");
    }
    if (input.visual.size() > 0 && calculateEmbeddingMagnitude(input.visual) > threat_threshold) {
        triggers.push_back("visual");
    }
    
    return triggers;
}

void MultiModalFusion::updateTemporalBuffer(const SensoryInput& input) {
    temporal_buffer_.push_back(input);
    
    // Remove old entries outside the integration window
    double current_time = input.timestamp;
    temporal_buffer_.erase(
        std::remove_if(temporal_buffer_.begin(), temporal_buffer_.end(),
            [current_time, this](const SensoryInput& buffered_input) {
                return (current_time - buffered_input.timestamp) > config_.temporal_integration_window;
            }),
        temporal_buffer_.end()
    );
}

Eigen::VectorXd MultiModalFusion::performTemporalIntegration(const std::vector<SensoryInput>& inputs) const {
    if (inputs.empty()) {
        return Eigen::VectorXd::Zero(512);
    }
    
    // Simple temporal integration: weighted average with recency bias
    Eigen::VectorXd integrated = Eigen::VectorXd::Zero(512);
    double total_weight = 0.0;
    
    for (size_t i = 0; i < inputs.size(); ++i) {
        double recency_weight = static_cast<double>(i + 1) / inputs.size(); // More recent = higher weight
        auto fused = performWeightedFusion(inputs[i]);
        
        if (fused.size() > 0) {
            if (integrated.size() != fused.size()) {
                integrated = Eigen::VectorXd::Zero(fused.size());
            }
            integrated += recency_weight * fused;
            total_weight += recency_weight;
        }
    }
    
    if (total_weight > 0.0) {
        integrated /= total_weight;
    }
    
    return integrated;
}

Eigen::VectorXd MultiModalFusion::normalizeEmbedding(const Eigen::VectorXd& embedding) const {
    double norm = embedding.norm();
    if (norm > 0.0) {
        return embedding / norm;
    }
    return embedding;
}

double MultiModalFusion::calculateEmbeddingMagnitude(const Eigen::VectorXd& embedding) const {
    return embedding.norm();
}

void MultiModalFusion::updateConfig(const FusionConfig& config) {
    config_ = config;
}

std::vector<MultiModalFusion::FusedRepresentation> MultiModalFusion::getFusionHistory() const {
    return fusion_history_;
}

void MultiModalFusion::clearHistory() {
    fusion_history_.clear();
    temporal_buffer_.clear();
}

} // namespace neurosim
