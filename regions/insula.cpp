#include "insula.hpp"

// Stub implementation for insula
// Owner: Darrell Mesa (darrell.mesa@pm-ss.org)

namespace neurosim {

Insula::Insula(const RegionConfig& region_config, 
              const InsulaConfig& insula_config)
    : BrainRegion(region_config), insula_config_(insula_config) {
}

double Insula::processInput(double input, double dt) {
    // Simple stub implementation
    current_activation_ = input * 0.6;
    return current_activation_;
}

} // namespace neurosim
