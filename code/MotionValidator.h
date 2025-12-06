// =======================================================================================
// MotionValidator.h
// Header for motion validation in Chance-Constrained RRT (CCRRT).
// Defines:
//   - StateKey: Helper for identifying states by position.
//   - CCRRTDetail::StateWithCovariance: Stores mean, covariance, and timestamp.
//   - UncertaintyManager: Propagates and stores uncertainty for each state.
//   - CCRRTMotionValidator: Checks if a motion is valid, considering:
//         1. Static obstacles (fixed in space)
//         2. Dynamic obstacles (move over time)
//         3. Chance constraints (probabilistic collision using uncertainty)
//
// State: 2D position (x, y)
// Control: 2D velocity (vx, vy)
// =======================================================================================

#ifndef MOTION_VALIDATOR_H
#define MOTION_VALIDATOR_H

#pragma once
#include <ompl/control/SpaceInformation.h>
#include <ompl/base/MotionValidator.h>
#include <ompl/base/SpaceInformation.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <Eigen/Dense>
#include <vector>
#include "Obstacles.h"

// Move these lines above any use of ob::State
namespace ob = ompl::base;
namespace oc = ompl::control;

// Full definition of StateKey
struct StateKey {
    double x, y;
    bool operator<(const StateKey& other) const {
        if (x != other.x) return x < other.x;
        return y < other.y;
    }
    static StateKey fromState(const ob::State* s) {
        auto* st = s->as<ob::RealVectorStateSpace::StateType>();
        return {st->values[0], st->values[1]};
    }
};

namespace CCRRTDetail {
    struct StateWithCovariance {
        Eigen::VectorXd mean;
        Eigen::MatrixXd covariance;
        double timestamp;
    };
}

// Manages state uncertainty propagation and chance constraint checking
class UncertaintyManager {
public:
    // Initialize with desired safety probability threshold
    UncertaintyManager(double psafe);

    // Store uncertainty (mean and covariance) for a given state
    void storeUncertainty(const ob::State* state,
                          const Eigen::VectorXd& mean, const Eigen::MatrixXd& cov, double timestamp);

    // Propagate uncertainty from one state to another using linear dynamics
    void propagateUncertainty(const ob::State* from, const ob::State* to,
                              const Eigen::MatrixXd& A, const Eigen::MatrixXd& B,
                              const Eigen::VectorXd& control, const Eigen::MatrixXd& Pw,
                              double deltaTime);

    // Mark satisfiesChanceConstraints as const
    bool satisfiesChanceConstraints(const ob::State* state,
                                   const std::vector<Obstacle>& obstacles) const;

    // Add timestamp to state uncertainty storage
    struct StateUncertainty {
        Eigen::VectorXd mean;
        Eigen::MatrixXd covariance;
        double timestamp;
    };

    // Add this getter method
    const std::map<const ob::State*, StateUncertainty>& getStateUncertainty() const;

private:
    // Evaluate chance constraint for circular obstacles
    bool isCircleConstraintSatisfied(const Eigen::Vector2d& mean,
                                    const Eigen::Matrix2d& cov, const Obstacle& obs) const;

    // Convert rectangular obstacles to circular for constraint checking
    bool isRectConstraintSatisfied(const Eigen::Vector2d& mean,
                                   const Eigen::Matrix2d& cov, const Obstacle& obs) const;

    // Helper function to get dynamic obstacle position at time t
    Eigen::Vector2d getDynamicObstaclePosition(const Obstacle& obs, double time) const;

    double psafe_;  // Probability threshold for safety constraints
    std::map<const ob::State*, StateUncertainty> stateUncertainty_;
};

// Motion validator that incorporates uncertainty in collision checking
class CCRRTMotionValidator : public ob::MotionValidator {
public:
    CCRRTMotionValidator(const ob::SpaceInformationPtr& si, double psafe);

    void setObstacles(const std::vector<Obstacle>& obstacles);

    void setStateUncertainty(std::map<StateKey, CCRRTDetail::StateWithCovariance>* stateUncertainty);

    bool checkMotion(const ob::State* s1, const ob::State* s2) const override;

    bool checkMotion(const ob::State* s1, const ob::State* s2,
        std::pair<ob::State*, double>& lastValid) const override;

private:
    double psafe_;
    std::vector<Obstacle> obstacles_;
    std::map<StateKey, CCRRTDetail::StateWithCovariance>* stateUncertainty_ = nullptr;

    bool isChanceConstraintSatisfied(const CCRRTDetail::StateWithCovariance& stateUnc,
                                     const Obstacle& obs) const;
};

#endif // MOTION_VALIDATOR_H