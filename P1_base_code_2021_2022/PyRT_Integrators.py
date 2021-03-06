from PyRT_Common import *
from random import randint


# -------------------------------------------------Integrator Classes
# the integrators also act like a scene class in that-
# it stores all the primitives that are to be ray traced.
class Integrator(ABC):
    # Initializer - creates object list
    def __init__(self, filename_, experiment_name=''):
        # self.primitives = []
        self.filename = filename_ + experiment_name
        self.env_map = None  # not initialized
        self.scene = None

    @abstractmethod
    def compute_color(self, ray):
        pass

    def add_environment_map(self, env_map_path):
        self.env_map = EnvironmentMap(env_map_path)

    def add_scene(self, scene):
        self.scene = scene

    def get_filename(self):
        return self.filename

    # Simple render loop: launches 1 ray per pixel
    def render(self):
        # YOU MUST CHANGE THIS METHOD IN ASSIGNMENTS 1.1 and 1.2:
        cam = self.scene.camera  # camera object
        # ray = Ray()

        print('Rendering Image: ' + self.get_filename())
        for x in range(0, cam.width):
            for y in range(0, cam.height):
                # pixel = GREEN
                # pixel = RGBColor(x/cam.width,y/cam.height,0) #Assigment 1.1
                direction = self.scene.camera.get_direction(x, y)
                origin = Vector3D(0, 0, 0)
                ray = Ray(origin, direction)
                pixel = self.compute_color(ray)
                self.scene.set_pixel(pixel, x, y)  # save pixel to pixel array
            progress = (x / cam.width) * 100
            print('\r\tProgress: ' + str(progress) + '%', end='')
        # save image to file
        print('\r\tProgress: 100% \n\t', end='')
        full_filename = self.get_filename()
        self.scene.save_image(full_filename)


class LazyIntegrator(Integrator):
    def __init__(self, filename_):
        super().__init__(filename_ + '_Intersection')

    def compute_color(self, ray):
        return BLACK


class IntersectionIntegrator(Integrator):

    def __init__(self, filename_):
        super().__init__(filename_ + '_Intersection')

    def compute_color(self, ray):
        # ASSIGNMENT 1.2: PUT YOUR CODE HERE
        if self.scene.any_hit(ray):
            return RED
        return BLACK


class DepthIntegrator(Integrator):

    def __init__(self, filename_, max_depth_=10):
        super().__init__(filename_ + '_Depth')
        self.max_depth = max_depth_

    def compute_color(self, ray):
        # ASSIGNMENT 1.3: PUT YOUR CODE HERE
        this_hit = self.scene.closest_hit(ray)
        if this_hit.has_hit:
            hit_distance = this_hit.hit_distance
            max_depth = 5.0
            color = max(1 - (hit_distance / max_depth), 0)
            return RGBColor(color.x, color.y, color.z)
        return BLACK


class NormalIntegrator(Integrator):

    def __init__(self, filename_):
        super().__init__(filename_ + '_Normal')

    def compute_color(self, ray):
        # ASSIGNMENT 1.3: PUT YOUR CODE HERE
        this_hit = self.scene.closest_hit(ray)
        if this_hit.has_hit:
            normal = this_hit.normal
            color = (normal + Vector3D(1, 1, 1)) / 2
            return RGBColor(color.x, color.y, color.z)
        return BLACK


class PhongIntegrator(Integrator):

    def __init__(self, filename_):
        super().__init__(filename_ + '_Phong')

    def compute_color(self, ray):
        # ASSIGNMENT 1.4: PUT YOUR CODE HERE
        Phong_color = RGBColor(0, 0, 0)
        hit_data = self.scene.closest_hit(ray)
        if hit_data.has_hit:
            # if the pixel obtain a light source
            # We take the light source,
            SourceLight = self.scene.pointLights[0]
            # Here, we obtain the position and the intensity from the source light
            positionLight = SourceLight.pos
            intensityLight = SourceLight.intensity

            # Euclidean distance between the point and the source light
            distance = positionLight - hit_data.hit_point
            direction_components = np.array((distance.x, distance.y, distance.z), dtype=np.float64)
            norm_distance = np.linalg.norm(direction_components)
            dir_distance = distance / norm_distance
            primitiva = self.scene.object_list[hit_data.primitive_index]
            ray_light = Ray(hit_data.hit_point, dir_distance, norm_distance)
            shadow_hit = self.scene.any_hit(ray_light)

            new_ray = Ray(hit_data, distance, dir_distance)

            if not shadow_hit:
                value = primitiva.BRDF.get_value(normal=hit_data.normal, wo=1, wi=dir_distance)

                # Equation of diffuse
                value = value.multiply(SourceLight.intensity / (norm_distance ** 2))
                return value + primitiva.BRDF.kd.multiply(self.scene.i_a)
            else:
                return primitiva.BRDF.kd.multiply(self.scene.i_a)
        else:
            return BLACK


class CMCIntegrator(Integrator):  # Classic Monte Carlo Integrator

    def __init__(self, n, filename_, experiment_name=''):
        filename_mc = filename_ + '_MC_' + str(n) + '_samples' + experiment_name
        super().__init__(filename_mc)
        self.n_samples = n
        self.uniform_pdf = UniformPDF()

    def compute_color(self, ray):

        hit_data = self.scene.closest_hit(ray)
        # Generate a sample set ???? of samples over the hemisphere

        if hit_data.has_hit:
            color = self.monte_carlo(hit_data, ray)
        else:
            if self.scene.env_map is not None:
                # ???????? (???????? ) = scene.env_map.getValue(???????? );
                color = self.scene.env_map.getValue(ray.d)
            else:
                color = BLACK
        return color

    def monte_carlo(self, hit_data, ray):
        primitiva = self.scene.object_list[hit_data.primitive_index]
        fr_material = primitiva.get_BRDF()
        normal_surf = hit_data.normal
        inverse_view_port = ray.d * -1

        samples, probabilities = sample_set_hemisphere(self.n_samples, self.uniform_pdf)
        rend = BLACK
        # For each sample ???????? ??? ????:
        for sample, probability in zip(samples, probabilities):

            # Center the sample around the surface normal, yielding ???????
            sample = center_around_normal(sample, normal_surf)
            #if hit_data.has_hit:

            # Create a secondary ray ???? with direction ???????????
            r = Ray(hit_data.hit_point, sample)
            # Shoot ???? by calling the method scene.closest_hit()
            shoot_r = self.scene.closest_hit(r)
            # If ???? hits the scene geometry, then:
            if shoot_r.has_hit:
                primitiva_two = self.scene.object_list[shoot_r.primitive_index]
                # ???????? (???????? ) = object_hit.emission;
                L_w = primitiva_two.emission
            else:
                if self.scene.env_map is not None:
                    # ???????? (???????? ) = scene.env_map.getValue(???????? );
                    L_w = self.scene.env_map.getValue(sample)
                else:
                    L_w = BLACK
            fr_gama = fr_material.get_value(sample, inverse_view_port, normal_surf)
            cos_gama = Dot(sample, normal_surf)
            rend += (L_w.multiply(fr_gama)* cos_gama)/probability
        result = rend / self.n_samples
        return result


class BayesianMonteCarloIntegrator(Integrator):
    def __init__(self, n, myGP, filename_, experiment_name=''):
        filename_bmc = filename_ + '_BMC_' + str(n) + '_samples' + experiment_name
        super().__init__(filename_bmc)
        self.n_samples = n
        self.myGP = myGP

    def compute_color(self, ray):
        GP_index = randint(0, len(self.myGP) - 1)
        estimate = BLACK
        this_hit = self.scene.closest_hit(ray)
        if this_hit.has_hit:
            object_brdf = self.scene.object_list[this_hit.primitive_index].get_BRDF()
            normal = this_hit.normal
            samples_pos = self.myGP[GP_index].samples_pos
            rotated_samples_pos = []
            rotated_samples_val = []
            # rotate samples around the normal of the ray hit and randomly rotate them
            for sample_pos in samples_pos:
                rotated_sample_pos = center_around_normal(rotate_around_y(random()*360,sample_pos), normal)
                rotated_samples_pos.append(rotated_sample_pos)

            # sample new values shooting the rotated samples positions from the hit_point
            for sample_idx, sample in enumerate(rotated_samples_pos):
                r = Ray(this_hit.hit_point, sample)
                # Shoot ???? by calling the method scene.closest_hit()
                shoot_r = self.scene.closest_hit(r)
                if shoot_r.has_hit:
                    primitiva_two = self.scene.object_list[shoot_r.primitive_index]
                    L_w = primitiva_two.emission
                    L_w = L_w.multiply(object_brdf.get_value(sample, 1, normal))
                else:
                    if self.scene.env_map is not None:
                        # ???????? (???????? ) = scene.env_map.getValue(???????? );

                        L_w = self.scene.env_map.getValue(sample)
                        L_w = L_w.multiply(object_brdf.get_value(sample, 1, normal))
                    else:
                        L_w = BLACK
                rotated_samples_val.append(L_w)

            self.myGP[GP_index].add_sample_val(rotated_samples_val)
            estimate = self.myGP[GP_index].compute_integral_BMC()
        elif self.scene.env_map is not None:
            estimate = self.scene.env_map.getValue(ray.d)
        else:
            estimate = BLACK

        return estimate
