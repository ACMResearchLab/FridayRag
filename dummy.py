from bs4 import BeautifulSoup

def function1(m):
    s = m.group(0)[2:].rstrip(';；')
    if s.startswith('x'):
        return chr(int('0'+s, 16))
    else:
        return chr(int(s))
    
def function2(lst):
    seen = set()
    result = []
    for item in lst:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

def function3(y):
    return [[0] * y[1] for _ in range(y[0])]

def function4(player, urls):
    import subprocess
    import shlex
    if (sys.version_info >= (3, 3)):
        import shutil
        exefile=shlex.split(player)[0]
        if shutil.which(exefile) is not None:
            subprocess.call(shlex.split(player) + list(urls))
        else:
            log.wtf('[Failed] Cannot find player "%s"' % exefile)
    else:
        subprocess.call(shlex.split(player) + list(urls))




def function5(data):
    normalized_data = {}
    for key, values in data.items():
        mean = sum(values) / len(values)
        std_dev = (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5
        normalized_data[key] = [(x - mean) / std_dev for x in values]
    return normalized_data




def function6(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    return soup.get_text()

def function7(self):
    # Derivation: https://sachinruk.github.io/blog/von-Mises-Fisher/
    event_dim = tf.compat.dimension_value(self.event_shape[0])
    if event_dim is None:
      raise ValueError('event shape must be statically known for _bessel_ive')
    safe_conc = tf.where(self.concentration > 0,
                         self.concentration, tf.ones_like(self.concentration))
    safe_mean = self.mean_direction * (
        _bessel_ive(event_dim / 2, safe_conc) /
        _bessel_ive(event_dim / 2 - 1, safe_conc))[..., tf.newaxis]
    return tf.where(
        self.concentration[..., tf.newaxis] > tf.zeros_like(safe_mean),
        safe_mean, tf.zeros_like(safe_mean))

def function8(x, y):
    n = len(x)
    x_mean = sum(x) / n
    y_mean = sum(y) / n
    numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
    denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
    slope = numerator / denominator
    intercept = y_mean - slope * x_mean
    return slope, intercept


def functionn0(self, subscription_id=None, cert_file=None,
                host=MANAGEMENT_HOST, request_session=None,
                timeout=DEFAULT_HTTP_TIMEOUT):
    '''
    Initializes the management service.

    subscription_id:
        Subscription to manage.
    cert_file:
        Path to .pem certificate file (httplib), or location of the
        certificate in your Personal certificate store (winhttp) in the
        CURRENT_USERCertificateName format.
        If a request_session is specified, then this is unused.
    host:
        Live ServiceClient URL. Defaults to Azure public cloud.
    request_session:
        Session object to use for http requests. If this is specified, it
        replaces the default use of httplib or winhttp. Also, the cert_file
        parameter is unused when a session is passed in.
        The session object handles authentication, and as such can support
        multiple types of authentication: .pem certificate, oauth.
        For example, you can pass in a Session instance from the requests
        library. To use .pem certificate authentication with requests
        library, set the path to the .pem file on the session.cert
        attribute.
    timeout:
        Optional. Timeout for the http request, in seconds.
    '''
    super(ServiceManagementService, self).__init__(
        subscription_id, cert_file, host, request_session, timeout)

def function21(numbers):
    sorted_numbers = sorted(numbers)
    n = len(sorted_numbers)
    if n % 2 == 0:
        return (sorted_numbers[n // 2 - 1] + sorted_numbers[n // 2]) / 2
    else:
        return sorted_numbers[n // 2]

def function22(html_text):
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html_text, 'html.parser')
    return soup.get_text()


def function2000(self, use_pandas=False):

    model = self._model_json["output"]
    if self.algo=='glm' or "variable_importances" in list(model.keys()) and model["variable_importances"]:
        if self.algo=='glm':
            tempvals = model["standardized_coefficient_magnitudes"].cell_values
            maxVal = 0
            sum=0
            for item in tempvals:
                sum=sum+item[1]
                if item[1]>maxVal:
                    maxVal = item[1]
            vals = []
            for item in tempvals:
                tempT = (item[0], item[1], item[1]/maxVal, item[1]/sum)
                vals.append(tempT)
            header = ["variable", "relative_importance", "scaled_importance", "percentage"]
        else:
            vals = model["variable_importances"].cell_values
            header = model["variable_importances"].col_header
            
        if use_pandas and can_use_pandas():
            import pandas
            return pandas.DataFrame(vals, columns=header)
        else:
            return vals
    else:
        print("Warning: This model doesn't have variable importances")

def function23(matrix, scalar):
    return [[element * scalar for element in row] for row in matrix]

def function24(word):
    return word == word[::-1]

def function254(data):
    cleaned_data = {}
    for key, values in data.items():
        mean = sum(values) / len(values) if len(values) != 0 else 0
        std_dev = (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5 if len(values) != 0 else 0
        cleaned_data[key] = [x for x in values if abs(x - mean) <= 2 * std_dev]
    return cleaned_data

def function26(dict1, dict2):
    merged_dict = dict1.copy()
    merged_dict.update(dict2)
    return merged_dict

def function27(point1, point2):
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5

def function28(sentence):
    return ' '.join(word.capitalize() for word in sentence.split())

def function29(lst):
    return lst[::-1]

def function30(url):
    import urllib.parse
    parsed_url = urllib.parse.urlparse(url)
    return parsed_url.netloc

def function31(nums):
    n = len(nums) + 1
    total_sum = n * (n + 1) // 2
    current_sum = sum(nums)
    return total_sum - current_sum