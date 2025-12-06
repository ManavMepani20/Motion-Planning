///////////////////////////////////////
// RBE 550 - Motion Planning
// Project 4: RG-RRT Implementation
// Authors: Pranay Katyal, Harsh Chhadej
//////////////////////////////////////

#include "RG-RRT.h"
#include <ompl/base/goals/GoalSampleableRegion.h>
#include <ompl/tools/config/SelfConfig.h>
#include <ompl/control/spaces/RealVectorControlSpace.h>
#include <limits>
// TODO: Implement RGRRT as described

ompl::control::RGRRT::RGRRT(const SpaceInformationPtr &si) : base::Planner(si, "RGRRT")
{
    specs_.approximateSolutions = true;
    siC_ = si.get();
    addIntermediateStates_ = false;
    goalBias_ = 0.05;
    number_of_control_inputs_ = 10;
    use_first_control_ = true;
    Planner::declareParam<double>("goal_bias", this, &RGRRT::setGoalBias, &RGRRT::getGoalBias, "0.:.05:1.");
    Planner::declareParam<bool>("intermediate_states", this, &RGRRT::setIntermediateStates, &RGRRT::getIntermediateStates);
    Planner::declareParam<int>("number_of_control_inputs", this, &RGRRT::setControlSampleSize, &RGRRT::getControlSampleSize, "2:1:10000");
    Planner::declareParam<bool>("first_control_sample", this, &RGRRT::setFirstControlSample, &RGRRT::getFirstControlSample); 
}

ompl::control::RGRRT::~RGRRT(void)
{
    freeMemory();
}

void ompl::control::RGRRT::clear(void)
{
    Planner::clear();
    sampler_.reset();
    controlSampler_.reset();
    freeMemory();
    // if (nn_)
    nn_->clear();
    lastGoalMotion_.reset();
}

void ompl::control::RGRRT::freeMemory(void)
{
    if (nn_)
    {
        std::vector<std::shared_ptr<Motion>> motions;
        nn_->list(motions);
        for (const auto &motion : motions)
        {
            si_->freeState(motion->state);
            siC_->freeControl(motion->control);
        }
    }
}

void ompl::control::RGRRT::setup(void)
{
    base::Planner::setup();
    if (!nn_)
        //NearestNeighborsLinear
        nn_ = std::make_shared<NearestNeighborsLinear<std::shared_ptr<Motion>>>();
    nn_->setDistanceFunction([this](const std::shared_ptr<Motion> &a, const std::shared_ptr<Motion> &b)
                             { return distanceFunction(a, b); });
}

void ompl::control::RGRRT::setupReachableSet(std::shared_ptr<Motion> m)
{
    // Ensure minimum of two control inputs (u_min and u_max)
    if(number_of_control_inputs_<2)
    {
        OMPL_WARN("%s: Invalid number_of_control_inputs. Setting default to 2. Must be greater than 2!", getName().c_str());
        number_of_control_inputs_ = 2;
    }

    // Calculate control step sizes
    const std::vector<double> diff = siC_->getControlSpace()->as<RealVectorControlSpace>()->getBounds().getDifference();
    for (double d : diff)
        control_step_.push_back(d / static_cast<double>(this->number_of_control_inputs_-1));

    // Get lower bounds of control space
    const std::vector<double> &low_bound = siC_->getControlSpace()->as<RealVectorControlSpace>()->getBounds().low;

    // Generate reachable motions
    for (int i = 0; i < this->number_of_control_inputs_; ++i)
    {
        auto motion = std::make_shared<Motion>(siC_);
        siC_->copyControl(motion->control, m->control);
        double *&controls = motion->control->as<RealVectorControlSpace::ControlType>()->values;
        controls[0] = low_bound[0] + control_step_[0] * (i);
        for (int j = 1; j < low_bound.size(); ++j)
        {
            if(use_first_control_)
                controls[j] = 0.0; //making second control as 0 
            else
                controls[j] = low_bound[j] + control_step_[j] * (i);
        }
        // Propagation step while valid
        motion->steps = siC_->propagateWhileValid(m->state, motion->control, siC_->getMinControlDuration(), motion->state);

        if (motion->steps != 0)
            m->reachable.push_back(motion);
    }
}

int ompl::control::RGRRT::selectReachableMotion(const std::shared_ptr<Motion> &qnear, const std::shared_ptr<Motion> &qrand)
{
    const double nearD = si_->distance(qnear->state, qrand->state);
    double minD = nearD;
    const auto &reachable = qnear->reachable;
    int id = -1;
    for (size_t i = 0; i < reachable.size(); ++i)
    {
        double newD = si_->distance(reachable[i]->state, qrand->state);
        if (newD < minD)
        {
            minD = newD;
            id = static_cast<int>(i);
        }
    }
    return id;
}

ompl::base::PlannerStatus ompl::control::RGRRT::solve(const base::PlannerTerminationCondition &ptc)
{
    checkValidity();
    base::Goal *goal = pdef_->getGoal().get();
    base::GoalSampleableRegion *goal_s = dynamic_cast<base::GoalSampleableRegion *>(goal);

    while (const base::State *st = pis_.nextStart())
    {
        auto motion = std::make_shared<Motion>(siC_);
        si_->copyState(motion->state, st);
        siC_->nullControl(motion->control);
        setupReachableSet(motion);
        nn_->add(motion);
    }

    if (nn_->size() == 0)
    {
        OMPL_ERROR("%s: There are no valid initial states!", getName().c_str());
        return base::PlannerStatus::INVALID_START;
    }

    if (!sampler_)
        sampler_ = si_->allocStateSampler();
    if (!controlSampler_)
        controlSampler_ = siC_->allocDirectedControlSampler();

    OMPL_INFORM("%s: Starting with %u states", getName().c_str(), nn_->size());

    std::shared_ptr<Motion> solution = nullptr;
    std::shared_ptr<Motion> approxsol = nullptr;
    double approxdif = std::numeric_limits<double>::infinity();

    auto rmotion = std::make_shared<Motion>(siC_);
    base::State *rstate = rmotion->state;
    Control *rctrl = rmotion->control;
    auto xstate = si_->allocState();

    while (ptc == false)
    {
        int id = -1;
        std::shared_ptr<Motion> nmotion = nullptr;

        /* sample random state (with goal biasing) */
        if (goal_s && rng_.uniform01() < goalBias_ && goal_s->canSample())
            goal_s->sampleGoal(rstate);
        else
            sampler_->sampleUniform(rstate);

        /* find closest state in the tree */
        nmotion = nn_->nearest(rmotion);
        /* select a reachable Motion*/
        id = selectReachableMotion(nmotion, rmotion);

            if(id!=-1)
            {

                auto motion = std::make_shared<Motion>(siC_);
                siC_->copyControl(motion->control,nmotion->reachable[id]->control);
                si_->copyState(motion->state,nmotion->reachable[id]->state);
                motion->steps = nmotion->reachable[id]->steps;
                motion->parent = nmotion;
                setupReachableSet(motion);
                nn_->add(motion);

                double dist = 0.0;
                bool solved = goal->isSatisfied(motion->state, &dist);
                if (solved)
                {
                    approxdif = dist;
                    solution = motion;
                    break;
                }
                if (dist < approxdif)
                {
                    approxdif = dist;
                    approxsol = motion;
                }
            }
        // }
    }

    bool solved = false;
    bool approximate = false;
    if (!solution)
    {
        solution = approxsol;
        approximate = true;
    }

    if (solution)
    {
        lastGoalMotion_ = solution;

        std::vector<std::shared_ptr<Motion>> mpath;
        while (solution)
        {
            mpath.push_back(solution);
            solution = solution->parent.lock();
        }

        auto path = std::make_shared<PathControl>(si_);
        for (int i = mpath.size() - 1; i >= 0; --i)
        {
            if (auto parent = mpath[i]->parent.lock())
                path->append(mpath[i]->state, mpath[i]->control, mpath[i]->steps * siC_->getPropagationStepSize());
            else
                path->append(mpath[i]->state);
        }
        solved = true;
        pdef_->addSolutionPath(base::PathPtr(path), approximate, approxdif);
    }

    si_->freeState(xstate);
    OMPL_INFORM("%s: Created %u states", getName().c_str(), nn_->size());

    return base::PlannerStatus(solved, approximate);
}

void ompl::control::RGRRT::getPlannerData(base::PlannerData &data) const
{
    Planner::getPlannerData(data);

    std::vector<std::shared_ptr<Motion>> motions;
    if (nn_)
        nn_->list(motions);

    double delta = siC_->getPropagationStepSize();

    if (lastGoalMotion_)
        data.addGoalVertex(base::PlannerDataVertex(lastGoalMotion_->state));

    for (const auto &m : motions)
    {
        if (auto parent = m->parent.lock())
        {
            if (data.hasControls())
                data.addEdge(base::PlannerDataVertex(parent->state), base::PlannerDataVertex(m->state), control::PlannerDataEdgeControl(m->control, m->steps * delta));
            else
                data.addEdge(base::PlannerDataVertex(parent->state), base::PlannerDataVertex(m->state));
        }
        else
        {
            data.addStartVertex(base::PlannerDataVertex(m->state));
        }
    }
}