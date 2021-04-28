/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using namespace std;
using string;
using vector;

// declaring a global random engine
static default_random_engine engine;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 100;  // TODO: Set the number of particles


  normal_distribution<double> dist_init_x(x, std[0]);
  normal_distribution<double> dist_init_y(y, std[1]);
  normal_distribution<double> dist_init_theta(theta, std[2]);
  //default_random_engine engine;

  // particles inizialization and random noise addition
  for (int i = 0; i < num_particles; i++) {
    Particle new_p;
    new_p.id     = i;
    new_p.x      = x;
    new_p.y      = y;
    new_p.theta  = theta;
    new_p.weight = 1.0;

    // add noise
    new_p.x     += dist_init_x(engine);
    new_p.y     += dist_init_y(engine);
    new_p.theta += dist_init_theta(engine);

    particles.push_back(new_p);
  }

  // initialization done
  is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */

  normal_distribution<double> dist_x(0, std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[1]);
  normal_distribution<double> dist_theta(0, std_pos[2]);

  for (int i = 0; i < num_particles; i++) {
    // calculate the new particles state
    if (fabs(yaw_rate) < 0.0001) {  
      particles[i].x += velocity * delta_t * cos(particles[i].theta);
      particles[i].y += velocity * delta_t * sin(particles[i].theta);
    } 
    else {
      particles[i].x      += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate*delta_t)  - sin(particles[i].theta));
      particles[i].y      += velocity / yaw_rate * (cos(particles[i].theta)                     - cos(particles[i].theta + yaw_rate * delta_t));
      particles[i].theta  += yaw_rate * delta_t;
    }

    // add noise
    particles[i].x      += dist_x(engine);
    particles[i].y      += dist_y(engine);
    particles[i].theta  += dist_theta(engine);

  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */

  for (int i = 0; i < observations.size(); i++) {
    // set the minimum distance to maximum possible limit
    double minimum_distance = numeric_limits<double>::max();
    
    for (int j = 0; j < predicted.size(); j++) {
      // set the distance between current and predicted landmarks
      double current_distance = dist(observations[i].x, observations[i].y, predicted[i].x, predicted[i].y);

      // find the predicted landmark nearest the current observed landmark
      if (minimum_distance > current_distance) {
        minimum_distance = current_distance;
        observations[i].id = predicted[i].id;
      }
    }
  }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], const vector<LandmarkObs> &observations, const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */

  for (int i = 0; i < num_particles; i++) {

    // computing the particle x, y and theta coordinates
    double particle_x     = particles[i].x;
    double particle_y     = particles[i].y;
    double particle_theta = particles[i].theta;
    
    // 1 --> computing valid landmarks
    // new vector containing the map landmark locations predicted to be within the sensor range of the particle
    vector<LandmarkObs> predictions;

    for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {

      // computing the landmark id, x and y coordinates
      double landmark_x = map_landmarks.landmark_list[j].x_f;
      double landmark_y = map_landmarks.landmark_list[j].y_f;
      int landmark_id   = map_landmarks.landmark_list[j].id_i;
      
      // computing distance between landmark and particle and considering only landmarks within sensor range of the particle
      if ( dist(particle_x, particle_y, landmark_x, landmark_y) <= sensor_range ) {

        // add prediction to vector
        predictions.push_back(LandmarkObs{ landmark_id, landmark_x, landmark_y });
      }
    }

    // 2 --> convert observations coordinates from vehicle to map
    vector<LandmarkObs> observations_map;
    double cos_theta = cos(particle_theta);
    double sin_theta = sin(particle_theta);

    for (int j = 0; j < observations.size(); j++) {
      double tmp_x = observations[j].x * cos_theta - observations[j].y * sin_theta + particle_x;
      double tmp_y = observations[j].x * sin_theta + observations[j].y * cos_theta + particle_y;
      //tmp.id = obs.id; // maybe an unnecessary step, since the each obersation will get the id from dataAssociation step.
      observations_map.push_back(LandmarkObs{ observations[j].id, tmp_x, tmp_y });
    }

    // 3 --> computing the landmark index for each observation
    dataAssociation(predictions, observations_map);

    // 3 --> computing the particle's weight:

    // weight re-initialization
    particles[i].weight = 1.0;

    for (int j = 0; j < observations_map.size(); j++) {

      // computing observation and associated prediction coordinates
      double observations_x, observations_y, predictions_x, predictions_y;
      observations_x = observations_map[j].x;
      observations_y = observations_map[j].y;

      int associated_prediction = observations_map[j].id;

      // computing the x and y coordinates of the prediction associated with the current observation
      for (int k = 0; k < predictions.size(); k++) {
        if (predictions[k].id == associated_prediction) {
          predictions_x = predictions[k].x;
          predictions_y = predictions[k].y;
        }
      }

      double x_term = pow(observations_x - predictions_x, 2) / (2 * pow(std_landmark[0], 2));
      double y_term = pow(observations_y - predictions_y, 2) / (2 * pow(std_landmark[1], 2));
      double w = exp(-(x_term + y_term)) / (2 * M_PI * std_landmark[0] * std_landmark[1]);
      particles[i].weight *=  w;
    }
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here. http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */

  vector<Particle> new_particles;
  new_particles.resize(num_particles);

  // get all the current weights
  vector<double> weights;
  for (int i = 0; i < num_particles; i++) {
    weights.push_back(particles[i].weight);
  }

  // compute a random starting index for resampling wheel
  discrete_distribution<> discrete_int_dist(weights.begin(), weights.end());
  int index = discrete_int_dist(engine);

  // compute the max weight
  double max_weight = *max_element(weights.begin(), weights.end());

  // compute a random distribution [0.0, max_weight)
  discrete_distribution<> discrete_real_dist(0.0, max_weight);

  double beta = 0.0;

  // spin the resample wheel!
  for (int i = 0; i < num_particles; i++) {
    beta += discrete_real_dist(engine) * 2.0;
    while (beta > weights[index]) {
      beta -= weights[index];
      index = (index + 1) % num_particles;
    }
    new_particles.push_back(particles[index]);
  }

  particles = new_particles;

}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}