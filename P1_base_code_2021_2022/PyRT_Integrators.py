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
            #si el pixel consigue fuente de luz
            #We take the light source,
            SourceLight = self.scene.pointLights[0]
            # Here, we obtain the position and the intensity from the source light
            positionLight = SourceLight.pos
            intensityLight = SourceLight.intensity



            #Euclidean distance between the point and the source light
            distance = positionLight - hit_data.hit_point
            direction_components = np.array((distance.x, distance.y, distance.z), dtype=np.float64)
            norm_distance = np.linalg.norm(direction_components)
            dir_distance = distance/norm_distance
            primitiva = self.scene.object_list[hit_data.primitive_index]
            ray_light = Ray(hit_data.hit_point,dir_distance,norm_distance)
            shadow_hit = self.scene.any_hit(ray_light)



            new_ray = Ray(hit_data,distance,dir_distance)


            if not shadow_hit:
                value = primitiva.BRDF.get_value(normal=hit_data.normal,wo=1,wi=dir_distance)

                #Equation of diffuse
                value = value.multiply(SourceLight.intensity/(norm_distance**2))
                return value + primitiva.BRDF.kd.multiply(self.scene.i_a)
            else:
                return primitiva.BRDF.kd.multiply(self.scene.i_a)
        else: return BLACK


class CMCIntegrator(Integrator):  # Classic Monte Carlo Integrator


    def __init__(self, n, filename_, experiment_name=''):
        filename_mc = filename_ + '_MC_' + str(n) + '_samples' + experiment_name
        super().__init__(filename_mc)
        self.n_samples = n

    def compute_color(self, ray):
        uniform_pdf = UniformPDF()
        hit_data = self.scene.closest_hit(ray)
        '''
        
        :param ray: 
        :return: 
        '''

        samples,probability = sample_set_hemisphere(self.n_samples, uniform_pdf)
        for self.n_samples in range(samples):

            # Center the sample around the surface normal, yielding 𝜔𝑗 ′
            new_ray = Ray()
            # Create a secondary ray 𝑟 with direction 𝜔𝑗 ′
            # Shoot 𝑟 by calling the method scene.closest_hit()
            if new_ray == True  :
                #𝐿𝑖 (𝜔𝑗 ) = object_hit.emission;

            else:
                if self.scene.env_map == :
                    #𝐿𝑖 (𝜔𝑗 ) = scene.env_map.getValue(𝜔𝑗 );
                End if



            pass


class BayesianMonteCarloIntegrator(Integrator):
    def __init__(self, n, myGP, filename_, experiment_name=''):
        filename_bmc = filename_ + '_BMC_' + str(n) + '_samples' + experiment_name
        super().__init__(filename_bmc)
        self.n_samples = n
        self.myGP = myGP

    def compute_color(self, ray):
        pass
