#pragma once
#include <Eigen/Dense>
#include <vector>

#ifndef OBSTACLES_H
#define OBSTACLES_H

#include <Eigen/Dense>

class Obstacle {
public:
    // Constructor for circular obstacle
    Obstacle(double x, double y, double radius) 
        : x_(x), y_(y), radius_(radius), isCircular_(true) {}
    
    // Constructor for rectangular obstacle
    Obstacle(double x, double y, double width, double height)
        : x_(x), y_(y), width_(width), height_(height), isCircular_(false) {}

    // Getters
    bool isCircular() const { return isCircular_; }
    double getX() const { return x_; }
    double getY() const { return y_; }
    double getRadius() const { return radius_; }
    double getWidth() const { return width_; }
    double getHeight() const { return height_; }
    
    Eigen::Vector2d getCenter() const {
        if (isCircular_) {
            return Eigen::Vector2d(x_, y_);
        }
        return Eigen::Vector2d(x_ + width_/2, y_ + height_/2);
    }

private:
    double x_, y_;        // Position
    double radius_;       // For circular obstacles
    double width_, height_; // For rectangular obstacles
    bool isCircular_;     // Type of obstacle
};

#endif // OBSTACLES_H

// Dynamic Obstacle that can move over time
class DynamicObstacle : public Obstacle {
public:
    DynamicObstacle(const Eigen::Vector2d& initialCenter, double radius, 
                   const Eigen::Vector2d& velocity, const Eigen::Vector2d& finalPosition)
        : Obstacle(initialCenter.x(), initialCenter.y(), radius),
          initialCenter_(initialCenter), 
          velocity_(velocity),
          finalPosition_(finalPosition) {}

    Eigen::Vector2d getPositionAtTime(double t) const {
        Eigen::Vector2d newPosition = initialCenter_ + velocity_ * t;
        if ((finalPosition_ - initialCenter_).dot(velocity_) > 0) {
            if ((newPosition - initialCenter_).dot(velocity_) > (finalPosition_ - initialCenter_).dot(velocity_)) {
                return finalPosition_;
            }
        }
        return newPosition;
    }

    // Add this method to provide position uncertainty
    double getPositionUncertainty() const {
        // Return a fixed uncertainty value (can be tuned)
        return 0.2;
    }

private:
    Eigen::Vector2d initialCenter_;
    Eigen::Vector2d velocity_;
    Eigen::Vector2d finalPosition_;
};

extern std::vector<Obstacle> staticObstacles;
extern std::vector<DynamicObstacle> dynamicObstacles;
