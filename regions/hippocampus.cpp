#include "hippocampus.hpp"

// Stub implementation for hippocampus
// Owner: Darrell Mesa (darrell.mesa@pm-ss.org)

namespace neurosim {

Hippocampus::Hippocampus(const RegionConfig& region_config, 
                        const HippocampusConfig& hippocampus_config)
    : BrainRegion(region_config), hippocampus_config_(hippocampus_config) {
}

double Hippocampus::processInput(double input, double dt) {
    // Simple stub implementation
    current_activation_ = input * 0.5;
    return current_activation_;
}

} // namespace neurosim
