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
            max_depth = 5
            depth_hit_ratio = hit_distance / max_depth
            color = max(1 - depth_hit_ratio, 0)
            return RGBColor(color, color, color)
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
        diffuse_contribution = RGBColor(0, 0, 0)
        specular_contribution = RGBColor(0, 0, 0)
        resulting_color = RGBColor(0, 0, 0)
        if self.scene.any_hit(ray):
            hit_data = self.scene.closest_hit(ray)
            hitted_object = self.scene.object_list[hit_data.primitive_index]
            # ambient contribution
            ambient_contribution = hitted_object.get_BRDF().kd.multiply(self.scene.i_a)

            # Check if pixel is lighted
            hit_point = hit_data.hit_point
            for pointLight in self.scene.pointLights:
                point_light_to_vector = pointLight.pos
                difference_hit_point_to_light = point_light_to_vector - hit_point
                difference_hit_point_to_light_norm = (difference_hit_point_to_light.x**2 + difference_hit_point_to_light.y**2 + difference_hit_point_to_light.z**2)**0.5
                difference_hit_point_to_light_normalized = difference_hit_point_to_light / difference_hit_point_to_light_norm
                difference_light_to_hit_poit_ray = Ray(hit_point, difference_hit_point_to_light_normalized, difference_hit_point_to_light_norm)
                difference_light_to_hit_poit_ray_hit_info = self.scene.closest_hit(difference_light_to_hit_poit_ray)

                if not difference_light_to_hit_poit_ray_hit_info.has_hit:
                    normal = hit_data.normal
                    diffuse_contribution = hitted_object.BRDF.get_value(normal=hit_data.normal, wo=1, wi=difference_hit_point_to_light_normalized)
                    diffuse_contribution = diffuse_contribution.multiply(pointLight.intensity/(difference_hit_point_to_light_norm**2))
                    # diffuse contribution
                    # diffuse_contribution = (pointLight.intensity/(difference_hit_point_to_light_norm**2))*kd*max(0, Dot(normal, difference_hit_point_to_light_normalized))
                    # specular contribution
                    viewer_vector_norm = (ray.d.x**2 + ray.d.y**2 + ray.d.z**2)**0.5
                    viewer_vector_normalized = ray.d / viewer_vector_norm
                    r = normal*2*(Dot(normal, difference_hit_point_to_light_normalized)) - difference_hit_point_to_light_normalized
                    K_s = 0.1
                    s = 25
                    specular_contribution = (pointLight.intensity/(difference_hit_point_to_light_norm**2))*K_s*max(0, Dot(Vector3D(0,0,0) - viewer_vector_normalized, r))**s
                    # specular_contribution = RGBColor(specular_contribution, specular_contribution, specular_contribution)
            if specular_contribution.r < 0 or specular_contribution.g < 0 or specular_contribution.b < 0 or \
                    specular_contribution.r > 1 or specular_contribution.g > 1 or specular_contribution.b > 1:
                print(specular_contribution)
            # ambient_contribution = RGBColor(0, 0, 0)
            # diffuse_contribution = RGBColor(0, 0, 0)
            # specular_contribution = RGBColor(0, 0, 0)
            resulting_color = ambient_contribution + diffuse_contribution + specular_contribution
        return resulting_color

class PhongIntegrator_(Integrator):

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
        # Generate a sample set 𝑆 of samples over the hemisphere

        if hit_data.has_hit:
            color = self.monte_carlo(hit_data, ray)
        else:
            if self.scene.env_map is not None:
                # 𝐿𝑖 (𝜔𝑗 ) = scene.env_map.getValue(𝜔𝑗 );
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
        # For each sample 𝜔𝑗 ∈ 𝑆:
        for sample, probability in zip(samples, probabilities):

            # Center the sample around the surface normal, yielding 𝜔�
            sample = center_around_normal(sample, normal_surf)
            #if hit_data.has_hit:

            # Create a secondary ray 𝑟 with direction 𝜔𝑗′
            r = Ray(hit_data.hit_point, sample)
            # Shoot 𝑟 by calling the method scene.closest_hit()
            shoot_r = self.scene.closest_hit(r)
            # If 𝑟 hits the scene geometry, then:
            if shoot_r.has_hit:
                primitiva_two = self.scene.object_list[shoot_r.primitive_index]
                # 𝐿𝑖 (𝜔𝑗 ) = object_hit.emission;
                L_w = primitiva_two.emission
            else:
                if self.scene.env_map is not None:
                    # 𝐿𝑖 (𝜔𝑗 ) = scene.env_map.getValue(𝜔𝑗 );
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
        pass
